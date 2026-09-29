import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getIrisData } from '../../api/datasets';

const SPECIES_COLORS: Record<string, string> = {
  'Iris setosa': '#4E79A7',
  'Iris versicolor': '#F28E2B',
  'Iris virginica': '#59A14F',
};

export default function Datasets() {
  const [data, setData] = useState<any>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchData = async (p: number) => {
    setLoading(true);
    try {
      const res = await getIrisData(p, 25);
      setData(res.data);
    } catch {
      setError('Failed to load dataset.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(page); }, [page]);

  const distributionData = data?.class_distribution
    ? Object.entries(data.class_distribution).map(([name, count]) => ({ name: name.replace('Iris ', ''), count, species: name }))
    : [];

  const totalPages = data ? Math.ceil(data.total / 25) : 1;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Iris Dataset</h1>
        <p className="page-subtitle">Fisher's Iris dataset — {data?.total || 150} records, 3 classes</p>
      </div>

      <div className="card">
        <h2 className="card-title">Class Distribution</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={distributionData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E9ECEF" />
            <XAxis dataKey="name" tick={{ fontSize: 13 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(val) => [`${val} samples`, 'Count']} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {distributionData.map((entry, idx) => (
                <Cell key={idx} fill={SPECIES_COLORS[entry.species] || '#4E79A7'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="card-header-row">
          <h2 className="card-title">Records</h2>
          <span className="text-muted">Page {page} of {totalPages}</span>
        </div>

        {loading ? (
          <div className="loading-spinner">Loading...</div>
        ) : error ? (
          <div className="alert-error">{error}</div>
        ) : (
          <>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Sepal Length</th>
                    <th>Sepal Width</th>
                    <th>Petal Length</th>
                    <th>Petal Width</th>
                    <th>Species</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.records?.map((r: any) => (
                    <tr key={r.id}>
                      <td>{r.id}</td>
                      <td>{r.sepal_length}</td>
                      <td>{r.sepal_width}</td>
                      <td>{r.petal_length}</td>
                      <td>{r.petal_width}</td>
                      <td>
                        <span className="species-badge" style={{ backgroundColor: `${SPECIES_COLORS[r.species]}22`, color: SPECIES_COLORS[r.species], border: `1px solid ${SPECIES_COLORS[r.species]}55` }}>
                          {r.species}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="pagination">
              <button className="btn-secondary btn-sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>← Prev</button>
              <span>{page} / {totalPages}</span>
              <button className="btn-secondary btn-sm" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next →</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
