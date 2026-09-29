import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isAdmin = user?.role === 'ADMIN';

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">🌸</span>
          <span className="logo-text">HMNC PRO</span>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">Navigation</div>

          <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">📊</span> Dashboard
          </NavLink>
          <NavLink to="/datasets" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">🗃️</span> Datasets
          </NavLink>
          <NavLink to="/quality" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">✅</span> Quality
          </NavLink>
          <NavLink to="/dictionary" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">📖</span> Dictionary
          </NavLink>
          <NavLink to="/prediction" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">🔮</span> Prediction Studio
          </NavLink>
          <NavLink to="/gallery" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">📷</span> Specimen Gallery
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">📋</span> Prediction History
          </NavLink>
          <NavLink to="/insights" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">💡</span> Model Insights
          </NavLink>
          <NavLink to="/benchmark" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">🔬</span> Model Benchmark
          </NavLink>

          {isAdmin && (
            <>
              <div className="nav-section-label admin-section-label">ML Administration</div>
              <NavLink to="/admin/svm-lab" className={({ isActive }) => `nav-link admin-link ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">⚗️</span> SVM Model Lab
              </NavLink>
              <NavLink to="/admin/evaluation" className={({ isActive }) => `nav-link admin-link ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">📈</span> Evaluation
              </NavLink>
              <NavLink to="/admin/decision-boundary" className={({ isActive }) => `nav-link admin-link ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">🗺️</span> Decision Boundary
              </NavLink>
              <NavLink to="/admin/model-history" className={({ isActive }) => `nav-link admin-link ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">📜</span> Model History
              </NavLink>
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="user-info">
            <span className="user-avatar">{user?.username?.charAt(0).toUpperCase()}</span>
            <div>
              <div className="user-name">{user?.username}</div>
              <div className="user-role-badge">{user?.role}</div>
            </div>
          </div>
          <button className="btn-logout" onClick={handleLogout}>Logout</button>
        </div>
      </aside>

      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
