import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { getModelInfo, trainModel } from '../../api/ml';

const CLASSIFIER_KEYS = ['rbf', 'linear', 'poly', 'sigmoid', 'mlp'];

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

export default function SVMModelLab() {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [infoLoading, setInfoLoading] = useState(true);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const fetchModelInfo = async () => {
    setInfoLoading(true);
    try {
      const res = await getModelInfo();
      setModelInfo(res.data);
    } catch {
      // Ignored if not yet trained
    } finally {
      setInfoLoading(false);
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const handleTrain = async () => {
    setError('');
    setMessage('');
    setLoading(true);
    try {
      const res = await trainModel();
      setTrainingResults(res.data);
      const bestName = CLASSIFIER_LABELS[res.data.best_kernel] || res.data.best_kernel?.toUpperCase();
      setMessage(`Huấn luyện thành công toàn bộ 5 mô hình! Mô hình đạt điểm đánh giá chéo cao nhất: ${bestName}.`);
      await fetchModelInfo();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Quá trình huấn luyện thất bại. Vui lòng kiểm tra lại hệ thống.');
    } finally {
      setLoading(false);
    }
  };

  // Construct chart data for all 5 classifiers if results are available
  const chartData = trainingResults?.results
    ? CLASSIFIER_KEYS.filter(k => trainingResults.results[k]).map(k => ({
        name: CLASSIFIER_LABELS[k] || k.toUpperCase(),
        '5-Fold CV Accuracy (%)': trainingResults.results[k]?.cv_mean
          ? Math.round(trainingResults.results[k].cv_mean * 1000) / 10
          : 0,
        'Test Accuracy (%)': trainingResults.results[k]?.test_accuracy
          ? Math.round(trainingResults.results[k].test_accuracy * 1000) / 10
          : 0,
      }))
    : [];

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">🧪 Phòng thí nghiệm mô hình (Model Lab Studio)</h1>
        <p className="page-subtitle">
          Trung tâm huấn luyện, tinh chỉnh tham số và so sánh hiệu năng song song giữa 4 hạt nhân SVM và mạng nơ-ron Deep Learning MLP.
        </p>
      </div>

      {/* Action / Trigger Card */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 className="card-title" style={{ marginBottom: '4px' }}>Huấn luyện lại toàn bộ mô hình (Full Re-training)</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Huấn luyện đồng thời 5 mô hình với phân chia ngẫu nhiên cố định (Stratified Split 80/20) và đánh giá chéo 5 phần (5-Fold CV).
            </p>
          </div>
          <button
            type="button"
            className="btn-primary"
            onClick={handleTrain}
            disabled={loading}
            style={{ minWidth: '220px', padding: '12px 24px', fontSize: '0.95rem' }}
          >
            {loading ? '⏳ Đang huấn luyện 5 mô hình...' : '🚀 Huấn luyện lại 5 mô hình'}
          </button>
        </div>

        {message && (
          <div className="alert-success" style={{ marginTop: '16px' }}>
            {message}
          </div>
        )}
        {error && (
          <div className="alert-error" style={{ marginTop: '16px' }}>
            {error}
          </div>
        )}
      </div>

      {/* Best Model Notice Banner */}
      {trainingResults?.best_kernel && (
        <div
          className="card"
          style={{
            marginBottom: '24px',
            backgroundColor: '#F0F8F0',
            borderLeft: '5px solid var(--success)',
            padding: '16px 20px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                Mô hình tối ưu theo Cross-Validation
              </div>
              <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--success)', marginTop: '2px' }}>
                ⭐ {CLASSIFIER_LABELS[trainingResults.best_kernel] || trainingResults.best_kernel}
                <span style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text)', marginLeft: '12px' }}>
                  (CV Score: {(trainingResults.cv_score * 100).toFixed(2)}%)
                </span>
              </div>
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', maxWidth: '420px', lineHeight: 1.4 }}>
              <strong>Lưu ý nghiệp vụ:</strong> Thông tin mô hình tối ưu chỉ mang ý nghĩa đối sánh và nghiên cứu. Trong Prediction Studio, người dùng hoàn toàn có quyền chủ động lựa chọn sử dụng bất kỳ mô hình nào trong 5 mô hình.
            </div>
          </div>
        </div>
      )}

      {/* Chart Section */}
      {chartData.length > 0 && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <h2 className="card-title">📊 Kết quả so sánh hiệu năng huấn luyện (5 Classifiers)</h2>
          <div style={{ width: '100%', height: 320, marginTop: '16px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[80, 100]} unit="%" tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value: any) => [`${value}%`]} />
                <Legend />
                <Bar dataKey="5-Fold CV Accuracy (%)" fill="#4E79A7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Test Accuracy (%)" fill="#59A14F" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Detailed Training Table */}
      {trainingResults?.results && (
        <div className="card" style={{ marginBottom: '24px', padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)' }}>
            <h2 className="card-title" style={{ margin: 0 }}>📋 Bảng chi tiết chỉ số kỹ thuật 5 mô hình</h2>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Mô hình phân loại</th>
                  <th>Loại kiến trúc</th>
                  <th>Test Accuracy</th>
                  <th>5-Fold CV Accuracy</th>
                  <th>Độ lệch CV (± Std)</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {CLASSIFIER_KEYS.filter(k => trainingResults.results[k]).map(k => {
                  const item = trainingResults.results[k];
                  const label = CLASSIFIER_LABELS[k];
                  const color = CLASSIFIER_COLORS[k];
                  const isBest = trainingResults.best_kernel === k;

                  return (
                    <tr key={k} style={isBest ? { backgroundColor: '#F9FFF9' } : {}}>
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
                          {label} {isBest ? '⭐' : ''}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                        {k === 'mlp' ? 'Deep Learning ANN' : 'Support Vector Machine'}
                      </td>
                      <td><strong>{(item.test_accuracy * 100).toFixed(2)}%</strong></td>
                      <td><strong>{(item.cv_mean * 100).toFixed(2)}%</strong></td>
                      <td className="text-muted">±{(item.cv_std * 100).toFixed(2)}%</td>
                      <td>{(item.precision * 100).toFixed(2)}%</td>
                      <td>{(item.recall * 100).toFixed(2)}%</td>
                      <td>{(item.f1 * 100).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Current Model Info Card */}
      {!infoLoading && modelInfo && (
        <div className="card">
          <h2 className="card-title">ℹ️ Thông tin trạng thái mô hình hiện hành</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginTop: '12px' }}>
            <div className="stat-card blue">
              <div className="stat-label">Mô hình tốt nhất hiện tại</div>
              <div className="stat-value" style={{ fontSize: '1.3rem', marginTop: '4px' }}>
                {CLASSIFIER_LABELS[modelInfo.kernel] || modelInfo.kernel_display || modelInfo.kernel}
              </div>
            </div>
            <div className="stat-card green">
              <div className="stat-label">Test Accuracy xác thực</div>
              <div className="stat-value" style={{ fontSize: '1.3rem', marginTop: '4px' }}>
                {modelInfo.test_accuracy ? (modelInfo.test_accuracy * 100).toFixed(2) + '%' : '—'}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">5-Fold CV Score</div>
              <div className="stat-value" style={{ fontSize: '1.3rem', marginTop: '4px' }}>
                {modelInfo.cv_mean ? (modelInfo.cv_mean * 100).toFixed(2) + '%' : '—'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
