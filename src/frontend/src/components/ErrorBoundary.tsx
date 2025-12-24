import type { ReactNode } from "react";
import { Component } from "react";

type ErrorBoundaryProps = {
  label?: string;
  children: ReactNode;
};

type ErrorBoundaryState = {
  hasError: boolean;
  error?: Error;
  info?: { componentStack?: string };
};

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string }) {
    this.setState({ error, info });
  }

  render() {
    const { hasError, error, info } = this.state;
    if (!hasError) return this.props.children;

    const label = this.props.label ?? "UI";
    const stack = import.meta.env.DEV ? info?.componentStack ?? "" : "";
    const message = error?.message ?? "Unknown error";
    const details = [`[${label}] ${message}`, stack].filter(Boolean).join("\n");

    return (
      <div className="card error-card">
        <div className="row-between">
          <h2>{label} crashed</h2>
          <span className="badge">error</span>
        </div>
        <div className="muted" style={{ marginTop: 6 }}>
          {message}
        </div>
        {stack ? (
          <pre className="error-stack">{stack}</pre>
        ) : (
          <div className="muted" style={{ marginTop: 6 }}>
            enable dev mode for stack trace
          </div>
        )}
        <div className="row error-actions">
          <button
            className="btn btn-ghost"
            onClick={() => {
              try {
                navigator.clipboard.writeText(details);
              } catch {}
            }}
          >
            Copy debug
          </button>
          <button
            className="btn"
            onClick={() => {
              this.setState({ hasError: false, error: undefined, info: undefined });
            }}
          >
            Retry render
          </button>
        </div>
      </div>
    );
  }
}
