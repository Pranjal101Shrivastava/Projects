import { Link, NavLink, Route, Routes, useParams } from "react-router-dom";
import { PROJECTS, bySlug } from "./lib/projects";
import Home from "./pages/Home";
import Methodology from "./pages/Methodology";
import ProjectPage from "./pages/ProjectPage";

function Nav() {
  return (
    <nav className="nav">
      <div className="wrap nav-inner">
        <Link to="/" className="nav-brand">
          <span className="accent-text">◈</span> Applied DS<span className="full"> Portfolio</span>
        </Link>
        <div className="nav-links">
          {PROJECTS.map((p) => (
            <NavLink key={p.slug} to={`/p/${p.slug}`}
                     className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
                     title={p.title}>
              {p.number}
            </NavLink>
          ))}
          <NavLink to="/methodology"
                   className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}>
            Methodology
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

function ProjectRoute() {
  const { slug } = useParams();
  const meta = slug ? bySlug(slug) : undefined;
  if (!meta) {
    return (
      <div className="wrap section">
        <h1>Not found</h1>
        <p className="dim">No project with that name. <Link to="/">Back to the index</Link>.</p>
      </div>
    );
  }
  return <ProjectPage meta={meta} />;
}

export default function App() {
  return (
    <div className="shell">
      <Nav />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/p/:slug" element={<ProjectRoute />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="wrap">
          <div className="row">
            <span>
              Eight end-to-end data science systems. Every figure is read from a committed
              JSON artifact produced by a pipeline in this repository.
            </span>
            <span className="spacer" />
            <a href="https://github.com/Pranjal101Shrivastava/Projects" target="_blank"
               rel="noreferrer noopener">Source</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
