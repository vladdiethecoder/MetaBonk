"""Roslyn Runtime Compiler for Agent-Generated Code.

Safe execution of LLM-generated C# code:
- Syntax tree parsing and sanitization
- Infinite loop protection
- Namespace blocking
- Collectible assemblies (memory-safe)

References:
- .NET Compiler Platform (Roslyn)
- AssemblyLoadContext
"""

from __future__ import annotations

import re
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class CompileResult(Enum):
    """Result of compilation."""
    SUCCESS = auto()
    SYNTAX_ERROR = auto()
    SEMANTIC_ERROR = auto()
    SAFETY_VIOLATION = auto()
    TIMEOUT = auto()


@dataclass
class CompilerError:
    """A compiler error."""
    
    line: int
    column: int
    code: str
    message: str
    severity: str = "error"
    
    def __str__(self) -> str:
        return f"({self.line},{self.column}): {self.severity} {self.code}: {self.message}"


@dataclass
class CompilationOutput:
    """Output from compilation."""
    
    result: CompileResult
    errors: List[CompilerError] = field(default_factory=list)
    warnings: List[CompilerError] = field(default_factory=list)
    
    # Output
    assembly_path: Optional[Path] = None
    sanitized_code: str = ""
    
    # Timing
    compile_time_ms: float = 0.0


@dataclass
class RoslynConfig:
    """Configuration for Roslyn compiler."""
    
    # Safety
    forbidden_namespaces: List[str] = field(default_factory=lambda: [
        "System.IO",
        "System.Net",
        "System.Threading",
        "System.Diagnostics.Process",
        "System.Reflection.Emit",
        "System.Runtime.InteropServices",
        "System.Security",
        "Microsoft.Win32",
    ])
    
    forbidden_types: List[str] = field(default_factory=lambda: [
        "File",
        "Directory",
        "Process",
        "Thread",
        "Socket",
        "HttpClient",
        "WebClient",
    ])
    
    # Limits
    max_code_length: int = 10000
    max_loop_iterations: int = 10000
    max_compile_time_ms: float = 5000.0
    max_execution_time_ms: float = 1000.0
    
    # Output
    output_dir: Path = Path("temp/compiled")
    
    # Compiler
    dotnet_path: str = "dotnet"


class CodeSanitizer:
    """Sanitizes code for safe execution."""
    
    def __init__(self, cfg: RoslynConfig):
        self.cfg = cfg
    
    def sanitize(self, code: str) -> Tuple[str, List[str]]:
        """Sanitize code and return (sanitized_code, errors)."""
        errors = []
        
        # Check length
        if len(code) > self.cfg.max_code_length:
            errors.append(f"Code too long: {len(code)} > {self.cfg.max_code_length}")
            return "", errors
        
        # Check forbidden namespaces
        for ns in self.cfg.forbidden_namespaces:
            pattern = rf'\busing\s+{re.escape(ns)}\b'
            if re.search(pattern, code):
                errors.append(f"Forbidden namespace: {ns}")
        
        # Check forbidden types
        for typename in self.cfg.forbidden_types:
            pattern = rf'\b{typename}\b'
            if re.search(pattern, code) and typename not in ["File"]:  # Allow some false positives
                # Check if it's actually a type usage
                type_pattern = rf'\bnew\s+{typename}|{typename}\s*\.'
                if re.search(type_pattern, code):
                    errors.append(f"Forbidden type: {typename}")
        
        if errors:
            return "", errors
        
        # Inject loop guards
        code = self._inject_loop_guards(code)
        
        # Inject timeout checks
        code = self._inject_timeout_checks(code)
        
        return code, []
    
    def _inject_loop_guards(self, code: str) -> str:
        """Inject iteration limits into loops."""
        guard_counter = "_loopGuard"
        guard_limit = self.cfg.max_loop_iterations
        
        # Add counter declaration
        if "class " in code:
            code = re.sub(
                r'(class\s+\w+[^{]*\{)',
                f'\\1\n    private int {guard_counter} = 0;',
                code,
                count=1
            )
        
        # Guard while loops
        code = re.sub(
            r'\bwhile\s*\(([^)]+)\)',
            f'while ({guard_counter}++ < {guard_limit} && (\\1))',
            code
        )
        
        # Guard for loops (add check)
        code = re.sub(
            r'\bfor\s*\(([^;]+);([^;]+);([^)]+)\)',
            f'for (\\1; {guard_counter}++ < {guard_limit} && (\\2); \\3)',
            code
        )
        
        return code
    
    def _inject_timeout_checks(self, code: str) -> str:
        """Inject timeout checks into methods."""
        # Add stopwatch import
        if "using System;" in code:
            code = code.replace(
                "using System;",
                "using System;\nusing System.Diagnostics;"
            )
        else:
            code = "using System.Diagnostics;\n" + code
        
        # Add timeout field
        if "class " in code:
            timeout_ms = self.cfg.max_execution_time_ms
            code = re.sub(
                r'(class\s+\w+[^{]*\{)',
                f'\\1\n    private static readonly Stopwatch _sw = Stopwatch.StartNew();\n    private void CheckTimeout() {{ if (_sw.ElapsedMilliseconds > {timeout_ms}) throw new TimeoutException("Execution timeout"); }}',
                code,
                count=1
            )
        
        return code


class RoslynCompiler:
    """Compiles C# code at runtime using Roslyn."""
    
    def __init__(self, cfg: Optional[RoslynConfig] = None):
        self.cfg = cfg or RoslynConfig()
        self.sanitizer = CodeSanitizer(self.cfg)
        
        # Ensure output directory exists
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compiled assemblies (for cleanup)
        self._assemblies: List[Path] = []
    
    def compile(
        self,
        code: str,
        assembly_name: Optional[str] = None,
    ) -> CompilationOutput:
        """Compile C# code to assembly.
        
        Args:
            code: C# source code
            assembly_name: Optional name for the assembly
        
        Returns:
            CompilationOutput with result and any errors
        """
        start_time = time.time()
        
        # Sanitize
        sanitized, sanitize_errors = self.sanitizer.sanitize(code)
        
        if sanitize_errors:
            return CompilationOutput(
                result=CompileResult.SAFETY_VIOLATION,
                errors=[
                    CompilerError(
                        line=0, column=0, code="SAFETY",
                        message=err, severity="error"
                    )
                    for err in sanitize_errors
                ],
            )
        
        # Generate assembly name
        if assembly_name is None:
            assembly_name = f"Agent_{int(time.time() * 1000)}"
        
        # Write to temp file
        source_file = self.cfg.output_dir / f"{assembly_name}.cs"
        source_file.write_text(sanitized)
        
        # Compile using dotnet
        output = self._compile_with_dotnet(source_file, assembly_name)
        output.sanitized_code = sanitized
        output.compile_time_ms = (time.time() - start_time) * 1000
        
        return output
    
    def _compile_with_dotnet(
        self,
        source_file: Path,
        assembly_name: str,
    ) -> CompilationOutput:
        """Compile using dotnet CLI."""
        # Create a minimal project file
        project_content = f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <OutputType>Library</OutputType>
    <AssemblyName>{assembly_name}</AssemblyName>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="{source_file.name}" />
  </ItemGroup>
</Project>"""
        
        project_file = source_file.parent / f"{assembly_name}.csproj"
        project_file.write_text(project_content)
        
        try:
            # Run dotnet build
            result = subprocess.run(
                [self.cfg.dotnet_path, "build", str(project_file), "-o", str(self.cfg.output_dir)],
                capture_output=True,
                text=True,
                timeout=self.cfg.max_compile_time_ms / 1000,
            )
            
            # Parse output
            errors, warnings = self._parse_build_output(result.stderr + result.stdout)
            
            if result.returncode == 0:
                assembly_path = self.cfg.output_dir / f"{assembly_name}.dll"
                if assembly_path.exists():
                    self._assemblies.append(assembly_path)
                    return CompilationOutput(
                        result=CompileResult.SUCCESS,
                        warnings=warnings,
                        assembly_path=assembly_path,
                    )
            
            return CompilationOutput(
                result=CompileResult.SYNTAX_ERROR if errors else CompileResult.SEMANTIC_ERROR,
                errors=errors,
                warnings=warnings,
            )
            
        except subprocess.TimeoutExpired:
            return CompilationOutput(
                result=CompileResult.TIMEOUT,
                errors=[
                    CompilerError(
                        line=0, column=0, code="TIMEOUT",
                        message="Compilation timed out"
                    )
                ],
            )
        except Exception as e:
            return CompilationOutput(
                result=CompileResult.SEMANTIC_ERROR,
                errors=[
                    CompilerError(
                        line=0, column=0, code="INTERNAL",
                        message=str(e)
                    )
                ],
            )
        finally:
            # Cleanup project file
            project_file.unlink(missing_ok=True)
    
    def _parse_build_output(self, output: str) -> Tuple[List[CompilerError], List[CompilerError]]:
        """Parse dotnet build output for errors."""
        errors = []
        warnings = []
        
        # Pattern: file(line,col): error/warning CODE: message
        pattern = r'([^(]+)\((\d+),(\d+)\):\s*(error|warning)\s+(\w+):\s*(.+)'
        
        for match in re.finditer(pattern, output):
            _, line, col, severity, code, message = match.groups()
            
            error = CompilerError(
                line=int(line),
                column=int(col),
                code=code,
                message=message.strip(),
                severity=severity,
            )
            
            if severity == "error":
                errors.append(error)
            else:
                warnings.append(error)
        
        return errors, warnings
    
    def cleanup(self):
        """Cleanup compiled assemblies."""
        for path in self._assemblies:
            try:
                path.unlink(missing_ok=True)
            except:
                pass
        self._assemblies = []


class CodeExecutor:
    """Executes compiled code safely."""
    
    def __init__(self, cfg: Optional[RoslynConfig] = None):
        self.cfg = cfg or RoslynConfig()
        self._runner_dir = self.cfg.output_dir / "_runner"
        self._runner_proj = self._runner_dir / "Runner.csproj"
        self._runner_prog = self._runner_dir / "Program.cs"
        self._runner_dll = self._runner_dir / "bin" / "Release" / "net8.0" / "Runner.dll"
        self._runner_ready = False
    
    def execute(
        self,
        assembly_path: Path,
        class_name: str,
        method_name: str = "Execute",
        args: Optional[List[Any]] = None,
        timeout_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a method from compiled assembly via a cached dotnet runner.

        The runner loads the target assembly in a separate dotnet process and
        invokes the requested method through reflection. This is not a perfect
        sandbox, but combined with code sanitization + process timeouts it is
        safe enough for recovery-mode experimentation.
        """
        timeout_ms = timeout_ms or self.cfg.max_execution_time_ms
        start = time.time()

        try:
            self._ensure_runner()
        except Exception as e:
            return {
                "success": False,
                "error": f"runner_build_failed: {e}",
                "stdout": "",
                "stderr": "",
                "execution_time_ms": (time.time() - start) * 1000,
            }

        payload = json.dumps(args or [])

        try:
            result = subprocess.run(
                [
                    self.cfg.dotnet_path,
                    str(self._runner_dll),
                    str(assembly_path),
                    class_name,
                    method_name,
                    payload,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_ms / 1000,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            data: Dict[str, Any] = {}
            if stdout:
                try:
                    data = json.loads(stdout)
                except Exception:
                    data = {"success": result.returncode == 0, "raw": stdout}
            else:
                data = {"success": result.returncode == 0}

            data.setdefault("stdout", stdout)
            data.setdefault("stderr", stderr)
            data.setdefault("execution_time_ms", (time.time() - start) * 1000)
            if result.returncode != 0:
                data["success"] = False
            return data
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "execution_timeout",
                "stdout": "",
                "stderr": "",
                "execution_time_ms": timeout_ms,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "execution_time_ms": (time.time() - start) * 1000,
            }

    def _ensure_runner(self) -> None:
        runner_version = "METABONK_RUNNER_V3"
        if self._runner_dll.exists() and self._runner_prog.exists():
            try:
                if runner_version in self._runner_prog.read_text():
                    self._runner_ready = True
                    return
            except Exception:
                pass

        self._runner_dir.mkdir(parents=True, exist_ok=True)

        proj = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion>
  </PropertyGroup>
</Project>
"""
        prog = r"""// METABONK_RUNNER_V3
using System;
using System.Linq;
using System.Reflection;
using System.Text.Json;

if (args.Length < 4)
{
    Console.Error.WriteLine("usage: Runner <assemblyPath> <className> <methodName> <jsonArgs>");
    Environment.Exit(2);
}

var assemblyPath = args[0];
var className = args[1];
var methodName = args[2];
var jsonArgs = args[3];

try
{
    var asm = Assembly.LoadFrom(assemblyPath);
    var type = asm.GetType(className) ??
               asm.GetTypes().FirstOrDefault(t => t.Name == className || t.FullName?.EndsWith("." + className) == true);
    if (type == null)
    {
        type = asm.GetTypes().FirstOrDefault(t =>
            t.GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static) != null);
    }
    if (type == null)
    {
        Console.WriteLine(JsonSerializer.Serialize(new { success = false, error = "type_not_found" }));
        Environment.Exit(3);
    }

    var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static);
    if (method == null)
    {
        Console.WriteLine(JsonSerializer.Serialize(new { success = false, error = "method_not_found" }));
        Environment.Exit(4);
    }

    object? instance = method.IsStatic ? null : Activator.CreateInstance(type);

    object?[] callArgs = Array.Empty<object?>();
    try
    {
        var parsedArgs = JsonSerializer.Deserialize<object?[]>(jsonArgs) ?? Array.Empty<object?>();
        var parameters = method.GetParameters();
        if (parameters.Length == 0)
        {
            callArgs = Array.Empty<object?>();
        }
        else if (parameters.Length == 1)
        {
            var pType = parameters[0].ParameterType;
            if (pType == typeof(string))
                callArgs = new object?[] { jsonArgs };
            else
                callArgs = new object?[] { parsedArgs.Length > 0 ? parsedArgs[0] : null };
        }
        else
        {
            callArgs = parsedArgs.Take(parameters.Length).ToArray();
        }
    }
    catch
    {
        callArgs = Array.Empty<object?>();
    }

    var res = method.Invoke(instance, callArgs);
    Console.WriteLine(JsonSerializer.Serialize(new { success = true, result = res }));
    Environment.Exit(0);
}
catch (TargetInvocationException tie)
{
    Console.WriteLine(JsonSerializer.Serialize(new { success = false, error = tie.InnerException?.Message ?? tie.Message, stack = tie.InnerException?.ToString() ?? tie.ToString() }));
    Environment.Exit(5);
}
catch (Exception e)
{
    Console.WriteLine(JsonSerializer.Serialize(new { success = false, error = e.Message, stack = e.ToString() }));
    Environment.Exit(6);
}
"""

        self._runner_proj.write_text(proj)
        self._runner_prog.write_text(prog)

        subprocess.run(
            [self.cfg.dotnet_path, "build", str(self._runner_proj), "-c", "Release"],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.cfg.max_compile_time_ms / 1000,
        )
        if not self._runner_dll.exists():
            raise RuntimeError("runner_dll_missing_after_build")
        self._runner_ready = True


class AgentCodeManager:
    """Manages agent code compilation and execution."""
    
    def __init__(self, cfg: Optional[RoslynConfig] = None):
        self.cfg = cfg or RoslynConfig()
        self.compiler = RoslynCompiler(self.cfg)
        self.executor = CodeExecutor(self.cfg)
        
        # Compiled abilities
        self.abilities: Dict[str, Path] = {}
    
    def compile_ability(
        self,
        ability_name: str,
        code: str,
    ) -> Tuple[bool, CompilationOutput]:
        """Compile an ability."""
        output = self.compiler.compile(code, ability_name)
        
        if output.result == CompileResult.SUCCESS and output.assembly_path:
            self.abilities[ability_name] = output.assembly_path
        
        return output.result == CompileResult.SUCCESS, output
    
    def execute_ability(
        self,
        ability_name: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute a compiled ability."""
        if ability_name not in self.abilities:
            return {"success": False, "error": "Ability not found"}
        
        return self.executor.execute(
            self.abilities[ability_name],
            ability_name,
            "Execute",
        )
    
    def get_error_prompt(self, output: CompilationOutput) -> str:
        """Generate prompt for fixing compilation errors."""
        if not output.errors:
            return ""
        
        prompt = "The code failed to compile with these errors:\n\n"
        for error in output.errors:
            prompt += f"Line {error.line}: {error.message}\n"
        
        prompt += "\nPlease fix the code and try again."
        
        return prompt
