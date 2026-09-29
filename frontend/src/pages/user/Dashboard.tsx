import { useEffect, useState } from 'react';
import { useAuth } from '../../auth/AuthContext';
import { getModelInfo } from '../../api/ml';

interface ModelInfo {
  loaded: boolean;
  kernel?: string;
  kernel_display?: string;
  cv_mean?: number;
  test_accuracy?: number;
  trained_at?: string;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then(res => setModelInfo(res.data))
      .catch(() => setModelInfo(null))
      .finally(() => setLoading(false));
  }, []);

  const isAdmin = user?.role === 'ADMIN';

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">Welcome back, <strong>{user?.username}</strong></p>
      </div>

      <div className="card-grid">
        <div className="stat-card">
          <div className="stat-value">150</div>
          <div className="stat-label">Dataset Records</div>
          <div className="stat-icon">🗃️</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">3</div>
          <div className="stat-label">Iris Classes</div>
          <div className="stat-icon">🌸</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">4</div>
          <div className="stat-label">Features</div>
          <div className="stat-icon">📐</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{isAdmin ? 'ADMIN' : 'USER'}</div>
          <div className="stat-label">Your Role</div>
          <div className="stat-icon">{isAdmin ? '🔑' : '👤'}</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '1.5rem' }}>
        <h2 className="card-title">Current Best SVM Model</h2>
        {loading ? (
          <div className="loading-spinner">Loading model info...</div>
        ) : modelInfo?.loaded ? (
          <div className="model-info-grid">
            <div className="model-info-item">
              <span className="model-info-label">Model Type</span>
              <span className="model-info-value">Support Vector Machine (SVM)</span>
            </div>
            <div className="model-info-item">
              <span className="model-info-label">Best Kernel</span>
              <span className="model-info-value kernel-badge">{modelInfo.kernel_display || modelInfo.kernel?.toUpperCase()}</span>
            </div>
            {modelInfo.cv_mean && (
              <div className="model-info-item">
                <span className="model-info-label">Cross-Validation Accuracy</span>
                <span className="model-info-value">{(modelInfo.cv_mean * 100).toFixed(2)}%</span>
              </div>
            )}
            {modelInfo.test_accuracy && (
              <div className="model-info-item">
                <span className="model-info-label">Test Accuracy</span>
                <span className="model-info-value">{(modelInfo.test_accuracy * 100).toFixed(2)}%</span>
              </div>
            )}
            <div className="model-info-item">
              <span className="model-info-label">Supported Kernels</span>
              <span className="model-info-value">RBF · Linear · Polynomial</span>
            </div>
          </div>
        ) : (
          <p className="text-muted">No model loaded yet. {isAdmin ? 'Go to SVM Model Lab to train.' : 'Contact administrator.'}</p>
        )}
      </div>

      {isAdmin && (
        <div className="card admin-info-card" style={{ marginTop: '1rem' }}>
          <h2 className="card-title">🔑 Admin Access</h2>
          <p>You have administrator privileges. Access the ML Administration section in the sidebar to:</p>
          <ul className="admin-feature-list">
            <li>⚗️ Train SVM models (RBF, Linear, Polynomial)</li>
            <li>📈 View detailed evaluation metrics</li>
            <li>🗺️ Visualize decision boundaries</li>
            <li>📜 Review model training history</li>
          </ul>
        </div>
      )}

      <div className="card" style={{ marginTop: '1rem' }}>
        <h2 className="card-title">Deep Learning Pipeline</h2>
        <div className="pipeline-stages">
          {['Data', 'Preprocessing', 'Model', 'Training', 'Evaluation', 'Prediction'].map((stage, i) => (
            <div key={stage} className="pipeline-stage">
              <div className="pipeline-step">{i + 1}</div>
              <div className="pipeline-label">{stage}</div>
              {i < 5 && <div className="pipeline-arrow">→</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
