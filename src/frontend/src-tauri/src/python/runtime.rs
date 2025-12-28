use std::path::PathBuf;

/// Minimal "Python runtime" wrapper for MetaBonk commands.
///
/// Note: MetaBonk currently runs its Python stack as subprocesses (see `scripts/`).
/// This module exists as the integration point referenced in docs/MetaBonk_Practical_Implementation.docx.
#[allow(dead_code)]
pub struct PythonRuntime {
  pub root: PathBuf,
  pub python_bin: String,
}

impl PythonRuntime {
  #[allow(dead_code)]
  pub fn new(root: PathBuf, python_bin: String) -> Self {
    Self { root, python_bin }
  }
}
