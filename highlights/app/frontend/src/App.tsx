import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { getUser } from "./api";
import Login from "./pages/Login";
import ProjectPage from "./pages/ProjectPage";
import Projects from "./pages/Projects";

function RequireUser() {
  return getUser() ? <Outlet /> : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<RequireUser />}>
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/projects" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
