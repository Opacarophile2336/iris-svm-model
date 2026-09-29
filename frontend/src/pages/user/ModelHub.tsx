import { useEffect, useState } from 'react';
import { getModelMetrics } from '../../api/prediction';
import { getModelInfo } from '../../api/ml';

const CLASSIFIER_KEYS = ['rbf', 'linear', 'poly', 'sigmoid', 'mlp'] as const;

const CLASSIFIER_LABELS: Record<string, string> = {
  rbf: 'SVM RBF',
  linear: 'SVM Linear',
  poly: 'SVM Polynomial',
  sigmoid: 'SVM Sigmoid',
  mlp: 'Deep Learning MLP',
};

const CLASSIFIER_COLORS: Record<string, string> = {
  rbf: '#4E79A7',
  linear: '#F28E2B',
  poly: '#59A14F',
  sigmoid: '#7970A1',
  mlp: '#9C27B0',
};

const SVM_KERNEL_MATH = {
  rbf: {
    title: 'Hạt nhân Radial Basis Function (SVM RBF)',
    formula: "K(x, x') = exp(-γ ||x - x'||²)",
    description:
      'Ánh xạ các vector đặc trưng đầu vào vào không gian Hilbert vô hạn chiều. Vượt trội trong việc phân tách các lớp phi tuyến tính phức tạp khi ranh giới tạo thành các đường bao khép kín.',
    gamma: 'scale',
    c: 1.0,
  },
  linear: {
    title: 'Hạt nhân Tuyến tính (SVM Linear)',
    formula: "K(x, x') = xᵀ x'",
    description:
      'Tính toán tích vô hướng trực tiếp trong không gian đặc trưng ban đầu. Tốc độ suy luận cực nhanh, kháng hiện tượng quá khớp (overfitting) mạnh mẽ và tối ưu hoàn hảo cho các lớp phân tách tuyến tính rõ rệt như Iris setosa.',
    gamma: 'N/A',
    c: 1.0,
  },
  poly: {
    title: 'Hạt nhân Đa thức bậc 3 (SVM Polynomial)',
    formula: "K(x, x') = (γ xᵀ x' + r)ᵈ",
    description:
      'Tính toán các tổ hợp đặc trưng đa thức lên tới bậc d=3, mô hình hóa các đường cong uốn lượn để phân biệt vùng giao thoa giữa Versicolor và Virginica.',
    gamma: 'scale',
    c: 1.0,
  },
  sigmoid: {
    title: 'Hạt nhân Sigmoid (SVM Sigmoid)',
    formula: "K(x, x') = tanh(γ xᵀ x' + r)",
    description:
      'Sử dụng hàm tiếp tuyến Hyperbolic Tangent (tanh) bắt nguồn từ lý thuyết mạng nơ-ron nhân tạo. Ánh xạ các điểm dữ liệu qua bề mặt chuyển tiếp sigmoid hình chữ S, hỗ trợ phân loại linh hoạt trong không gian đặc trưng đối xứng.',
    gamma: 'scale',
    c: 1.0,
  },
};

export default function ModelHub() {
  const [metrics, setMetrics] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getModelMetrics(), getModelInfo()])
      .then(([metricsRes, infoRes]) => {
        setMetrics(metricsRes.data);
        setModelInfo(infoRes.data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">🔬 Trung tâm đối chuẩn mô hình (Model Hub & Benchmark)</h1>
        <p className="page-subtitle">
          Tổng quan kiến trúc toán học, bảng đối chuẩn hiệu năng thực tế và cơ sở lý thuyết của 5 bộ phân loại trong IrisAI Studio.
        </p>
      </div>

      {/* System Runtime Status */}
      <div className="card-grid">
        <div className="stat-card green">
          <div className="stat-value">Port 2006</div>
          <div className="stat-label">FastAPI Backend (Hoạt động)</div>
        </div>
        <div className="stat-card blue">
          <div className="stat-value">Port 2336</div>
          <div className="stat-label">Vite Frontend (Hoạt động)</div>
        </div>
        <div className="stat-card" style={{ borderLeft: '3px solid #9C27B0' }}>
          <div className="stat-value">5 Classifiers</div>
          <div className="stat-label">4 SVM Kernels + 1 Deep Learning MLP</div>
        </div>
        <div className="stat-card green">
          <div className="stat-value">
            {modelInfo?.kernel ? CLASSIFIER_LABELS[modelInfo.kernel] || modelInfo.kernel_display : 'SVM Linear'}
          </div>
          <div className="stat-label">Mô hình tối ưu 5-Fold CV</div>
        </div>
      </div>

      {/* Real Validated Metrics Comparison Table */}
      <div className="card" style={{ marginTop: '1.2rem' }}>
        <h2 className="card-title">📊 Bảng đối chuẩn hiệu năng thực tế (Stratified Test Set 20%)</h2>
        <p className="text-muted" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>
          Tất cả các chỉ số hiệu năng bên dưới được trích xuất trực tiếp từ các mô hình đã được huấn luyện trên 150 mẫu Iris kinh điển, đánh giá độc lập trên tập kiểm định 30 mẫu và 5-Fold Cross-Validation.
        </p>

        {loading ? (
          <div className="loading-spinner">Đang tải dữ liệu đối chuẩn 5 mô hình...</div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Mô hình phân loại</th>
                  <th>Họ kiến trúc</th>
                  <th>Test Accuracy</th>
                  <th>5-Fold CV Accuracy</th>
                  <th>Độ lệch chuẩn CV</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {CLASSIFIER_KEYS.map((k) => {
                  const m = metrics?.[k];
                  const isOptimal = k === (modelInfo?.kernel || 'linear');
                  const label = CLASSIFIER_LABELS[k];
                  const color = CLASSIFIER_COLORS[k];

                  return (
                    <tr key={k} className={isOptimal ? 'best-row' : ''}>
                      <td>
                        <span
                          className="kernel-badge"
                          style={{
                            backgroundColor: `${color}18`,
                            color: color,
                            border: `1px solid ${color}44`,
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '4px',
                          }}
                        >
                          {label}
                        </span>
                        {isOptimal && ' 🏆 (Top CV)'}
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                        {k === 'mlp' ? 'Deep Learning Neural Net' : 'Support Vector Machine'}
                      </td>
                      <td><strong>{m?.test_accuracy ? `${m.test_accuracy}%` : '—'}</strong></td>
                      <td><strong>{m?.cv_accuracy ? `${m.cv_accuracy}%` : '—'}</strong></td>
                      <td className="text-muted">{m?.cv_std !== undefined ? `±${m.cv_std}%` : '—'}</td>
                      <td>{m?.precision ? `${m.precision}%` : '—'}</td>
                      <td>{m?.recall ? `${m.recall}%` : '—'}</td>
                      <td><strong>{m?.f1 ? `${m.f1}%` : '—'}</strong></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Mathematical Formulations of the 4 SVM Kernels */}
      <div className="card" style={{ marginTop: '1.2rem' }}>
        <h2 className="card-title">📐 Cơ sở toán học 4 hạt nhân Support Vector Machine (SVM)</h2>
        <div className="kernel-math-grid">
          {(['rbf', 'linear', 'poly', 'sigmoid'] as const).map((k) => {
            const math = SVM_KERNEL_MATH[k];
            return (
              <div key={k} className="kernel-math-card" style={{ borderTop: `3px solid ${CLASSIFIER_COLORS[k]}` }}>
                <h3 className="kernel-math-title" style={{ color: CLASSIFIER_COLORS[k] }}>{math.title}</h3>
                <div className="kernel-formula-box">
                  <code>{math.formula}</code>
                </div>
                <p className="kernel-math-desc">{math.description}</p>
                <div className="kernel-param-list">
                  <span>Tham số điều hòa C: <code>{math.c}</code></span>
                  <span>Tham số Gamma: <code>{math.gamma}</code></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Deep Learning Architecture Details Card */}
      <div className="card" style={{ marginTop: '1.2rem', borderTop: '4px solid #9C27B0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h2 className="card-title" style={{ color: '#9C27B0', marginBottom: '2px' }}>
              🧠 Kiến trúc mạng nơ-ron Deep Learning MLP (Multi-Layer Perceptron)
            </h2>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Phân loại dựa trên mô hình học sâu truyền thẳng (Feed-Forward Artificial Neural Network)
            </span>
          </div>
          <span
            style={{
              backgroundColor: '#9C27B018',
              color: '#9C27B0',
              border: '1px solid #9C27B044',
              padding: '4px 10px',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '0.8rem',
            }}
          >
            Deep Learning Framework: Scikit-Learn MLPClassifier
          </span>
        </div>

        <div className="kernel-formula-box" style={{ background: '#FAF5FF', borderColor: '#E9D8FD', padding: '12px 16px' }}>
          <code style={{ color: '#6B46C1', fontSize: '0.92rem', fontWeight: 600 }}>
            Input (4 Features: SL, SW, PL, PW) → Dense(64, ReLU) → Dense(32, ReLU) → Output Dense(3, Softmax)
          </code>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px', marginTop: '16px' }}>
          <div style={{ background: '#F8F9FA', padding: '12px 14px', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Tầng ẩn (Hidden Layers)</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>2 tầng ẩn (64 nơ-ron & 32 nơ-ron)</div>
            <small style={{ color: 'var(--text-muted)' }}>Hàm kích hoạt phi tuyến ReLU</small>
          </div>

          <div style={{ background: '#F8F9FA', padding: '12px 14px', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Thuật toán tối ưu (Optimizer)</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>Adam (Adaptive Moment Estimation)</div>
            <small style={{ color: 'var(--text-muted)' }}>Learning rate khởi tạo $\alpha = 0.001$</small>
          </div>

          <div style={{ background: '#F8F9FA', padding: '12px 14px', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Hàm mất mát (Loss Function)</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>Categorical Cross-Entropy (Log-Loss)</div>
            <small style={{ color: 'var(--text-muted)' }}>Phân phối xác suất chuẩn hóa đa lớp</small>
          </div>

          <div style={{ background: '#F8F9FA', padding: '12px 14px', borderRadius: '4px', border: '1px solid var(--border-light)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Tham số điều hòa (Regularization)</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>L2 Regularization ($\alpha = 0.0001$)</div>
            <small style={{ color: 'var(--text-muted)' }}>Early stopping & max 500 epochs</small>
          </div>
        </div>
      </div>

      {/* Botanical ML Principles Card */}
      <div className="card" style={{ marginTop: '1.2rem', borderLeft: '3px solid var(--primary)' }}>
        <h2 className="card-title">🌿 Ghi chú thực vật học & Khuyến nghị chọn mô hình</h2>
        <ul className="admin-feature-list" style={{ lineHeight: 1.6, fontSize: '0.88rem' }}>
          <li style={{ marginBottom: '8px' }}>
            <strong>Iris setosa:</strong> Có kích thước cánh hoa nhỏ đặc thù và hoàn toàn tách biệt tuyến tính trong không gian cánh hoa (Petal Length & Petal Width). Cả <strong>SVM Linear</strong>, <strong>SVM RBF</strong> và <strong>MLP</strong> đều đạt độ chính xác 100% khi phân loại loài này.
          </li>
          <li style={{ marginBottom: '8px' }}>
            <strong>Iris versicolor và Iris virginica:</strong> Có sự giao thoa nhẹ ở vùng chiều dài cánh hoa từ 4.5 đến 5.1 cm. Các mô hình phi tuyến tính như <strong>SVM RBF</strong>, <strong>SVM Polynomial (bậc 3)</strong> và <strong>Deep Learning MLP</strong> thể hiện khả năng uốn cong linh hoạt của siêu phẳng để nắm bắt các trường hợp biên tốt hơn.
          </li>
          <li>
            <strong>SVM Sigmoid:</strong> Đóng vai trò cầu nối lý thuyết quan trọng giữa SVM và mạng nơ-ron Perceptron cổ điển, hoàn thiện phổ nghiên cứu học máy từ mô hình siêu phẳng hình học đến mô hình xác suất học sâu.
          </li>
        </ul>
      </div>
    </div>
  );
}
