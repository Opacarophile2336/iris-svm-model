import { useEffect, useState } from 'react';
import { getModelHistory } from '../../api/ml';

const CLASSIFIER_DISPLAY: Record<string, string> = {
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

export default function ModelHistory() {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getModelHistory()
      .then(res => setHistory(Array.isArray(res.data) ? res.data.reverse() : []))
      .catch(() => setError('Không thể tải dữ liệu lịch sử huấn luyện.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page">
        <div className="loading-spinner">Đang tải lịch sử huấn luyện từ cơ sở dữ liệu...</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📜 Lịch sử huấn luyện mô hình (Model History)</h1>
        <p className="page-subtitle">
          Nhật ký các phiên huấn luyện và đánh giá mô hình được lưu trữ trực tiếp trên cơ sở dữ liệu SQL Server. Tổng số {history.length} phiên đã ghi nhận.
        </p>
      </div>

      {error && <div className="alert-error" style={{ marginBottom: '16px' }}>{error}</div>}

      {history.length === 0 ? (
        <div className="card empty-state" style={{ textAlign: 'center', padding: '48px 20px' }}>
          <div className="empty-icon" style={{ fontSize: '2.5rem', marginBottom: '12px' }}>📜</div>
          <h3 style={{ marginBottom: '8px' }}>Chưa có phiên huấn luyện nào</h3>
          <p style={{ color: 'var(--text-muted)' }}>
            Hãy sử dụng <strong>Model Lab Studio</strong> để huấn luyện mô hình và lưu trữ lịch sử đánh giá.
          </p>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Mã phiên</th>
                  <th>Phân loại</th>
                  <th>Mô hình tối ưu</th>
                  <th>Thời gian huấn luyện</th>
                  <th>Điểm CV tối ưu</th>
                  <th>Test Accuracy</th>
                  <th>Các mô hình đã đối sánh</th>
                  <th>Trạng thái</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => {
                  const kernelKey = (h.best_kernel || h.Kernel || '').toLowerCase();
                  const color = CLASSIFIER_COLORS[kernelKey] || '#4E79A7';
                  const label =
                    h.kernel_display ||
                    CLASSIFIER_DISPLAY[kernelKey] ||
                    (h.best_kernel ? h.best_kernel.toUpperCase() : 'SVM');

                  const trainingDateStr = h.training_date || h.CreatedAt || h.created_at;
                  const cvScore = h.cv_score ?? h.Accuracy;
                  const testAcc = h.test_accuracy ?? h.Accuracy;

                  return (
                    <tr key={h.model_id || h.ModelID}>
                      <td><span className="history-id">#{h.model_id || h.ModelID}</span></td>
                      <td>
                        <strong>{h.model_type || h.ModelName || 'SVM'}</strong>
                      </td>
                      <td>
                        <span
                          className="kernel-badge"
                          style={{
                            color: color,
                            backgroundColor: `${color}14`,
                            border: `1px solid ${color}44`,
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '4px',
                          }}
                        >
                          ⭐ {label}
                        </span>
                      </td>
                      <td className="text-muted" style={{ whiteSpace: 'nowrap' }}>
                        {trainingDateStr ? new Date(trainingDateStr).toLocaleString('vi-VN') : '—'}
                      </td>
                      <td>
                        <strong>{cvScore ? (cvScore * 100).toFixed(2) + '%' : '—'}</strong>
                      </td>
                      <td>
                        {testAcc ? (testAcc * 100).toFixed(2) + '%' : '—'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {h.kernels_compared ? (
                            Object.keys(h.kernels_compared).map((k) => {
                              const kColor = CLASSIFIER_COLORS[k] || '#4E79A7';
                              const kLabel = CLASSIFIER_DISPLAY[k] || k.toUpperCase();
                              const cvMean = h.kernels_compared[k]?.cv_mean;
                              return (
                                <span
                                  key={k}
                                  style={{
                                    fontSize: '0.75rem',
                                    color: kColor,
                                    backgroundColor: `${kColor}10`,
                                    border: `1px solid ${kColor}33`,
                                    padding: '2px 6px',
                                    borderRadius: '3px',
                                    whiteSpace: 'nowrap',
                                  }}
                                >
                                  {kLabel}: {cvMean ? (cvMean * 100).toFixed(1) + '%' : '—'}
                                </span>
                              );
                            })
                          ) : (
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                              {label} (Đơn lẻ)
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <span className={`status-badge ${h.status === 'active' || !h.status ? 'active' : 'inactive'}`}>
                          {h.status || 'Active'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
