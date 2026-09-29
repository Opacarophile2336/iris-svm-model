import { useState } from 'react';
import { getDecisionBoundary } from '../../api/ml';

const FEATURE_OPTIONS = [
  { value: 0, label: 'Đài hoa: Chiều dài (Sepal Length)' },
  { value: 1, label: 'Đài hoa: Chiều rộng (Sepal Width)' },
  { value: 2, label: 'Cánh hoa: Chiều dài (Petal Length)' },
  { value: 3, label: 'Cánh hoa: Chiều rộng (Petal Width)' },
];

const KERNEL_OPTIONS = [
  { value: 'rbf', label: 'SVM RBF (Radial Basis Function)' },
  { value: 'linear', label: 'SVM Linear (Siêu phẳng tuyến tính)' },
  { value: 'poly', label: 'SVM Polynomial (Đa thức bậc 3)' },
  { value: 'sigmoid', label: 'SVM Sigmoid (Hàm Hyperbolic Tangent)' },
];

export default function DecisionBoundary() {
  const [kernel, setKernel] = useState('rbf');
  const [featureX, setFeatureX] = useState(2);
  const [featureY, setFeatureY] = useState(3);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    setError('');
    setImageB64(null);
    setLoading(true);
    try {
      const res = await getDecisionBoundary(kernel, featureX, featureY);
      setImageB64(res.data.image_base64);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Không thể tạo hình ảnh ranh giới quyết định.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">🗺️ Ranh giới quyết định 2D (Decision Boundary)</h1>
        <p className="page-subtitle">
          Trực quan hóa bề mặt phân tách lớp của các hạt nhân Support Vector Machine (SVM) trên không gian 2 chiều.
        </p>
      </div>

      {/* Scientific Architecture Notice */}
      <div
        className="card"
        style={{
          marginBottom: '20px',
          backgroundColor: '#F8F9FA',
          borderLeft: '4px solid var(--primary)',
          padding: '14px 18px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
          <span style={{ fontSize: '1.2rem', marginTop: '2px' }}>ℹ️</span>
          <div style={{ fontSize: '0.85rem', color: 'var(--text)', lineHeight: 1.5 }}>
            <strong>Phạm vi kiến trúc toán học:</strong> Công cụ trực quan hóa ranh giới quyết định (Decision Boundary) được thiết kế đặc thù cho các hàm nhân Support Vector Machine (gồm <strong>SVM RBF, SVM Linear, SVM Polynomial, SVM Sigmoid</strong>) thông qua lưới tọa độ phẳng Meshgrid.
            <br />
            Mô hình <strong>Deep Learning MLP</strong> hoạt động trên không gian ánh xạ phi tuyến đa tầng ẩn (64 $\rightarrow$ 32 nơ-ron), được đánh giá toàn diện qua Ma trận nhầm lẫn và Đồ thị hiệu năng tại trang <em>Model Evaluation</em>.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '24px' }}>
        <h2 className="card-title">Cấu hình tham số trực quan hóa</h2>
        <div className="boundary-controls" style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'flex-end' }}>
          <div className="form-group" style={{ minWidth: '260px', flex: 1, margin: 0 }}>
            <label className="form-label">Hạt nhân SVM (SVM Kernel)</label>
            <select
              className="form-input"
              value={kernel}
              onChange={e => setKernel(e.target.value)}
            >
              {KERNEL_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ minWidth: '220px', flex: 1, margin: 0 }}>
            <label className="form-label">Đặc trưng trục hoành (X-Axis)</label>
            <select
              className="form-input"
              value={featureX}
              onChange={e => setFeatureX(Number(e.target.value))}
            >
              {FEATURE_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value} disabled={opt.value === featureY}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ minWidth: '220px', flex: 1, margin: 0 }}>
            <label className="form-label">Đặc trưng trục tung (Y-Axis)</label>
            <select
              className="form-input"
              value={featureY}
              onChange={e => setFeatureY(Number(e.target.value))}
            >
              {FEATURE_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value} disabled={opt.value === featureX}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            className="btn-primary"
            onClick={handleGenerate}
            disabled={loading}
            style={{ minWidth: '160px', height: '40px' }}
          >
            {loading ? '⏳ Đang tạo đồ thị...' : '🎨 Vẽ ranh giới'}
          </button>
        </div>
      </div>

      {error && <div className="alert-error" style={{ marginBottom: '20px' }}>{error}</div>}

      {/* Decision Boundary Canvas Card */}
      <div className="card" style={{ textAlign: 'center', padding: '24px' }}>
        {loading ? (
          <div style={{ padding: '60px 0' }}>
            <div className="loading-spinner">Đang sinh lưới tọa độ và phân tách ranh giới cho {kernel.toUpperCase()}...</div>
          </div>
        ) : imageB64 ? (
          <div>
            <img
              src={`data:image/png;base64,${imageB64}`}
              alt={`Ranh giới quyết định SVM ${kernel.toUpperCase()}`}
              style={{
                maxWidth: '100%',
                maxHeight: '620px',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                border: '1px solid var(--border-light)',
              }}
            />
            <p style={{ marginTop: '12px', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Đồ thị hiển thị các vùng quyết định của 3 loài hoa (Setosa, Versicolor, Virginica) và vị trí của các điểm dữ liệu mẫu thực tế.
            </p>
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '48px 0' }}>
            <div className="empty-icon" style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🗺️</div>
            <h3 style={{ marginBottom: '6px' }}>Chưa có đồ thị ranh giới</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
              Hãy chọn hạt nhân SVM và cặp đặc trưng mong muốn, sau đó bấm <strong>"Vẽ ranh giới"</strong> để xem bề mặt phân lớp.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
