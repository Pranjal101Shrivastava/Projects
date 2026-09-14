import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";
import "./styles.css";

// HashRouter rather than BrowserRouter: GitHub Pages serves static files with no
// server-side rewrite, so a deep link like /Projects/p/fraud would 404 on refresh.
// Hash routing keeps every route reachable from a cold load.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>
);
