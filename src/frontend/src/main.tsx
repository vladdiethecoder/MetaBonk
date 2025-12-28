import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import RuntimeProvider from "./tauri/RuntimeProvider";
import "./styles.css";

const qc = new QueryClient();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={qc}>
    <RuntimeProvider>
      <BrowserRouter>
        <ErrorBoundary label="App">
          <App />
        </ErrorBoundary>
      </BrowserRouter>
      <Toaster theme="dark" position="bottom-right" richColors closeButton />
    </RuntimeProvider>
  </QueryClientProvider>,
);
