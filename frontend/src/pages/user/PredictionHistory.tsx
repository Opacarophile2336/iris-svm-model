import { useEffect, useState, useCallback } from 'react';
import {
  getHistory,
  getAllHistory,
  deletePrediction,
  clearHistory,
  exportHistoryBlob,
} from '../../api/prediction';
import type {
  PredictionHistoryItem,
  HistoryFilterParams,
  PaginatedHistoryResponse,
} from '../../api/prediction';
import { useAuth } from '../../auth/AuthContext';

const SPECIES_COLORS: Record<string, string> = {
  'Iris setosa': '#4E79A7',
  'Iris versicolor': '#F28E2B',
  'Iris virginica': '#59A14F',
};

const KERNEL_DISPLAY_MAP: Record<string, string> = {
  rbf: 'SVM RBF',
  linear: 'SVM Linear',
  poly: 'SVM Polynomial',
  sigmoid: 'SVM Sigmoid',
  mlp: 'Deep Learning MLP',
};

const KERNEL_COLORS: Record<string, string> = {
  rbf: '#4E79A7',
  linear: '#F28E2B',
  poly: '#59A14F',
  sigmoid: '#7970A1',
  mlp: '#9C27B0',
};

export default function PredictionHistory() {
  const { user } = useAuth();
  const isAdmin = user?.role?.toUpperCase() === 'ADMIN';

  // Active view tab: 'my' for user's personal history, 'admin' for system audit
  const [activeTab, setActiveTab] = useState<'my' | 'admin'>('my');

  // History list and pagination state
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalPages, setTotalPages] = useState(1);

  // Filters
  const [speciesFilter, setSpeciesFilter] = useState('');
  const [kernelFilter, setKernelFilter] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [searchUser, setSearchUser] = useState('');

  // UI state
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // Modal states
  const [deleteTarget, setDeleteTarget] = useState<PredictionHistoryItem | null>(null);
  const [isClearModalOpen, setIsClearModalOpen] = useState(false);

  const loadData = useCallback(() => {
    setLoading(true);
    setError('');

    const params: HistoryFilterParams = {
      page,
      page_size: pageSize,
      species: speciesFilter || undefined,
      kernel: kernelFilter || undefined,
      start_date: startDate ? new Date(startDate).toISOString() : undefined,
      end_date: endDate ? new Date(endDate + 'T23:59:59').toISOString() : undefined,
    };

    if (activeTab === 'admin' && isAdmin) {
      if (searchUser.trim()) {
        params.search = searchUser.trim();
      }
      getAllHistory(params)
        .then((res) => {
          const data = res.data as PaginatedHistoryResponse;
          if (data && 'items' in data) {
            setHistory(data.items);
            setTotal(data.total);
            setTotalPages(data.total_pages);
          } else if (Array.isArray(res.data)) {
            setHistory(res.data);
            setTotal(res.data.length);
            setTotalPages(1);
          }
        })
        .catch((err) => {
          setError(err?.response?.data?.detail || 'Không thể tải dữ liệu kiểm toán hệ thống.');
        })
        .finally(() => setLoading(false));
    } else {
      getHistory(params)
        .then((res) => {
          const data = res.data as PaginatedHistoryResponse;
          if (data && 'items' in data) {
            setHistory(data.items);
            setTotal(data.total);
            setTotalPages(data.total_pages);
          } else if (Array.isArray(res.data)) {
            setHistory(res.data);
            setTotal(res.data.length);
            setTotalPages(1);
          }
        })
        .catch((err) => {
          setError(err?.response?.data?.detail || 'Không thể tải lịch sử dự đoán cá nhân.');
        })
        .finally(() => setLoading(false));
    }
  }, [activeTab, isAdmin, page, pageSize, speciesFilter, kernelFilter, startDate, endDate, searchUser]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle Tab Change
  const handleTabChange = (tab: 'my' | 'admin') => {
    setActiveTab(tab);
    setPage(1);
    setSearchUser('');
  };

  // Handle Reset Filters
  const handleResetFilters = () => {
    setSpeciesFilter('');
    setKernelFilter('');
    setStartDate('');
    setEndDate('');
    setSearchUser('');
    setPage(1);
  };

  // Handle Export File
  const handleExport = async (format: 'xlsx' | 'csv') => {
    try {
      setExporting(true);
      setError('');
      const params: HistoryFilterParams = {
        species: speciesFilter || undefined,
        kernel: kernelFilter || undefined,
        start_date: startDate ? new Date(startDate).toISOString() : undefined,
        end_date: endDate ? new Date(endDate + 'T23:59:59').toISOString() : undefined,
        search: activeTab === 'admin' ? (searchUser.trim() || undefined) : undefined,
      };

      const res = await exportHistoryBlob(format, params);
      const mimeType =
        format === 'xlsx'
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : 'text/csv;charset=utf-8;';
      const blob = new Blob([res.data], { type: mimeType });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const dateStr = new Date().toISOString().slice(0, 10);
      const filePrefix = activeTab === 'admin' ? 'iris_admin_audit' : 'iris_prediction_history';
      link.setAttribute('download', `${filePrefix}_${dateStr}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      setSuccessMsg(`Xuất báo cáo ${format.toUpperCase()} thành công!`);
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err: any) {
      setError('Xuất dữ liệu thất bại. Vui lòng kiểm tra lại kết nối.');
    } finally {
      setExporting(false);
    }
  };

  // Handle Delete Single Record
  const confirmDeleteSingle = async () => {
    if (!deleteTarget) return;
    try {
      setActionLoading(true);
      setError('');
      await deletePrediction(deleteTarget.id);
      setSuccessMsg(`Đã xóa thành công bản ghi dự đoán #${deleteTarget.id}.`);
      setDeleteTarget(null);
      loadData();
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err: any) {
      setError(err?.response?.data?.detail || `Không thể xóa bản ghi #${deleteTarget.id}.`);
    } finally {
      setActionLoading(false);
    }
  };

  // Handle Clear User History
  const confirmClearHistory = async () => {
    try {
      setActionLoading(true);
      setError('');
      const res = await clearHistory();
      setSuccessMsg(res.data?.message || 'Đã xóa toàn bộ lịch sử dự đoán cá nhân của bạn.');
      setIsClearModalOpen(false);
      setPage(1);
      loadData();
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Không thể xóa lịch sử dự đoán.');
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="page">
      {/* Page Header */}
      <div className="page-header">
        <h1 className="page-title">
          {activeTab === 'admin' ? '🛡️ Kiểm toán toàn bộ dự đoán hệ thống' : '📋 Lịch sử dự đoán cá nhân'}
        </h1>
        <p className="page-subtitle">
          {activeTab === 'admin'
            ? 'Theo dõi, tra cứu người dùng, phân tích và xuất dữ liệu kiểm toán toàn bộ hệ thống IrisAI Studio.'
            : 'Quản lý, tra cứu, xuất dữ liệu và xem lại các kết quả dự đoán hoa Iris đã thực hiện.'}
        </p>
      </div>

      {/* Role Navigation Tabs if Admin */}
      {isAdmin && (
        <div className="audit-tabs">
          <button
            type="button"
            className={`audit-tab-btn ${activeTab === 'my' ? 'active' : ''}`}
            onClick={() => handleTabChange('my')}
          >
            👤 Lịch sử của tôi
          </button>
          <button
            type="button"
            className={`audit-tab-btn ${activeTab === 'admin' ? 'active' : ''}`}
            onClick={() => handleTabChange('admin')}
          >
            🛡️ Kiểm toán hệ thống (Admin)
          </button>
        </div>
      )}

      {/* Alerts */}
      {error && <div className="alert-error" style={{ marginBottom: '16px' }}>{error}</div>}
      {successMsg && <div className="alert-success" style={{ marginBottom: '16px' }}>{successMsg}</div>}

      {/* Filter and Search Bar */}
      <div className="filter-toolbar">
        {activeTab === 'admin' && (
          <div className="filter-item" style={{ minWidth: '180px' }}>
            <label>Tìm kiếm người dùng</label>
            <input
              type="text"
              placeholder="Nhập tên tài khoản..."
              value={searchUser}
              onChange={(e) => {
                setSearchUser(e.target.value);
                setPage(1);
              }}
            />
          </div>
        )}

        <div className="filter-item">
          <label>Loài hoa (Species)</label>
          <select
            value={speciesFilter}
            onChange={(e) => {
              setSpeciesFilter(e.target.value);
              setPage(1);
            }}
          >
            <option value="">Tất cả loài hoa</option>
            <option value="setosa">Iris setosa</option>
            <option value="versicolor">Iris versicolor</option>
            <option value="virginica">Iris virginica</option>
          </select>
        </div>

        <div className="filter-item">
          <label>Mô hình (Model / Kernel)</label>
          <select
            value={kernelFilter}
            onChange={(e) => {
              setKernelFilter(e.target.value);
              setPage(1);
            }}
          >
            <option value="">Tất cả mô hình</option>
            <option value="rbf">SVM RBF</option>
            <option value="linear">SVM Linear</option>
            <option value="poly">SVM Polynomial</option>
            <option value="sigmoid">SVM Sigmoid</option>
            <option value="mlp">Deep Learning MLP</option>
          </select>
        </div>

        <div className="filter-item">
          <label>Từ ngày</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => {
              setStartDate(e.target.value);
              setPage(1);
            }}
          />
        </div>

        <div className="filter-item">
          <label>Đến ngày</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => {
              setEndDate(e.target.value);
              setPage(1);
            }}
          />
        </div>

        <div className="filter-item" style={{ alignSelf: 'flex-end' }}>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={handleResetFilters}
            title="Xóa tất cả các điều kiện lọc"
          >
            🔄 Đặt lại bộ lọc
          </button>
        </div>
      </div>

      {/* Header Actions Row (Export & Clear) */}
      <div className="header-actions-row">
        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Hiển thị <strong>{history.length}</strong> / <strong>{total}</strong> bản ghi
          {activeTab === 'admin' ? ' (Toàn hệ thống)' : ' (Cá nhân)'}
        </div>

        <div className="export-group">
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={() => handleExport('xlsx')}
            disabled={exporting || total === 0}
            title="Xuất dữ liệu sang định dạng Excel (.xlsx)"
          >
            {exporting ? '⏳ Đang xuất...' : '📥 Xuất Excel (.xlsx)'}
          </button>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={() => handleExport('csv')}
            disabled={exporting || total === 0}
            title="Xuất dữ liệu sang định dạng CSV (.csv)"
          >
            {exporting ? '⏳ Đang xuất...' : '📄 Xuất CSV (.csv)'}
          </button>

          {activeTab === 'my' && (
            <button
              type="button"
              className="btn-outline-danger btn-sm"
              onClick={() => setIsClearModalOpen(true)}
              disabled={total === 0 || loading}
              style={{ marginLeft: '8px' }}
              title="Xóa toàn bộ lịch sử dự đoán của tài khoản này"
            >
              🗑️ Xóa toàn bộ lịch sử
            </button>
          )}
        </div>
      </div>

      {/* Content Table / Empty State */}
      {loading ? (
        <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
          <div className="loading-spinner">Đang tải dữ liệu lịch sử...</div>
        </div>
      ) : history.length === 0 ? (
        <div className="card empty-state" style={{ textAlign: 'center', padding: '48px 20px' }}>
          <div className="empty-icon" style={{ fontSize: '2.5rem', marginBottom: '12px' }}>📋</div>
          <h3 style={{ marginBottom: '8px' }}>Chưa có bản ghi dự đoán nào</h3>
          <p style={{ color: 'var(--text-muted)' }}>
            {speciesFilter || kernelFilter || startDate || endDate || searchUser
              ? 'Không tìm thấy kết quả phù hợp với bộ lọc hiện tại. Hãy thử đặt lại bộ lọc.'
              : 'Hãy sử dụng Prediction Studio để thực hiện dự đoán đầu tiên.'}
          </p>
        </div>
      ) : (
        <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {activeTab === 'admin' && <th>Người dùng</th>}
                  <th>Thời gian</th>
                  <th>Sepal L</th>
                  <th>Sepal W</th>
                  <th>Petal L</th>
                  <th>Petal W</th>
                  <th>Loài dự đoán</th>
                  <th>Độ tin cậy</th>
                  <th>Mô hình</th>
                  <th>Kernel</th>
                  <th style={{ textAlign: 'center' }}>Thao tác</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r) => {
                  const kernelKey = (r.kernel || '').toLowerCase();
                  const kernelDisplayName =
                    r.kernel_display || KERNEL_DISPLAY_MAP[kernelKey] || (r.kernel ? r.kernel.toUpperCase() : 'SVM');
                  const kernelColor = KERNEL_COLORS[kernelKey] || '#4E79A7';
                  const speciesColor = SPECIES_COLORS[r.predicted_species] || '#4E79A7';

                  return (
                    <tr key={r.id}>
                      <td><span className="history-id">#{r.id}</span></td>
                      {activeTab === 'admin' && (
                        <td>
                          <strong>{r.username || `User #${r.user_id}`}</strong>
                        </td>
                      )}
                      <td className="text-muted" style={{ whiteSpace: 'nowrap' }}>
                        {new Date(r.timestamp).toLocaleString('vi-VN')}
                      </td>
                      <td>{r.features?.sepal_length} cm</td>
                      <td>{r.features?.sepal_width} cm</td>
                      <td>{r.features?.petal_length} cm</td>
                      <td>{r.features?.petal_width} cm</td>
                      <td>
                        <span
                          className="species-badge"
                          style={{
                            backgroundColor: `${speciesColor}22`,
                            color: speciesColor,
                            border: `1px solid ${speciesColor}55`,
                            fontWeight: 600,
                          }}
                        >
                          {r.predicted_species}
                        </span>
                      </td>
                      <td>
                        <strong style={{ color: r.confidence >= 90 ? 'var(--success)' : 'inherit' }}>
                          {r.confidence}%
                        </strong>
                      </td>
                      <td>{r.model}</td>
                      <td>
                        <span
                          className="kernel-badge"
                          style={{
                            backgroundColor: `${kernelColor}18`,
                            color: kernelColor,
                            border: `1px solid ${kernelColor}44`,
                            padding: '3px 8px',
                            borderRadius: '4px',
                            fontWeight: 600,
                            fontSize: '0.78rem',
                          }}
                        >
                          {kernelDisplayName}
                        </span>
                      </td>
                      <td style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                        <button
                          type="button"
                          className="btn-outline-danger"
                          onClick={() => setDeleteTarget(r)}
                          title={`Xóa bản ghi #${r.id}`}
                        >
                          🗑️ Xóa
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination Footer */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px 20px',
              borderTop: '1px solid var(--border-light)',
              flexWrap: 'wrap',
              gap: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Bản ghi mỗi trang:</span>
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
                style={{
                  padding: '4px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                  fontSize: '0.82rem',
                }}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>

            <div className="pagination" style={{ margin: 0 }}>
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                ◀ Trang trước
              </button>
              <span>
                Trang <strong>{page}</strong> / <strong>{totalPages || 1}</strong>
              </span>
              <button
                type="button"
                className="btn-secondary btn-sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Trang sau ▶
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Delete Single Prediction */}
      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => !actionLoading && setDeleteTarget(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span style={{ fontSize: '1.4rem' }}>⚠️</span>
              <h3 className="modal-title">Xác nhận xóa bản ghi</h3>
            </div>
            <div className="modal-body">
              Bạn có chắc chắn muốn xóa bản ghi dự đoán <strong>#{deleteTarget.id}</strong> (
              <em>{deleteTarget.predicted_species}</em> - {deleteTarget.kernel_display || deleteTarget.kernel})?
              <br />
              <small style={{ color: 'var(--text-muted)', display: 'block', marginTop: '6px' }}>
                Hành động này sẽ xóa vĩnh viễn bản ghi khỏi cơ sở dữ liệu và không thể hoàn tác.
              </small>
            </div>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setDeleteTarget(null)}
                disabled={actionLoading}
              >
                Hủy bỏ
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={confirmDeleteSingle}
                disabled={actionLoading}
              >
                {actionLoading ? 'Đang xóa...' : 'Xóa bản ghi'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Clear User History */}
      {isClearModalOpen && (
        <div className="modal-backdrop" onClick={() => !actionLoading && setIsClearModalOpen(false)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span style={{ fontSize: '1.4rem' }}>🚨</span>
              <h3 className="modal-title">Xác nhận xóa toàn bộ lịch sử</h3>
            </div>
            <div className="modal-body">
              Bạn có chắc chắn muốn xóa <strong>toàn bộ lịch sử dự đoán cá nhân</strong> của mình không?
              <br />
              <small style={{ color: 'var(--danger)', fontWeight: 600, display: 'block', marginTop: '8px' }}>
                Cảnh báo: Toàn bộ {total} bản ghi dự đoán của bạn sẽ bị xóa vĩnh viễn và không thể khôi phục!
              </small>
            </div>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setIsClearModalOpen(false)}
                disabled={actionLoading}
              >
                Hủy bỏ
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={confirmClearHistory}
                disabled={actionLoading}
              >
                {actionLoading ? 'Đang xóa...' : 'Xác nhận xóa tất cả'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
