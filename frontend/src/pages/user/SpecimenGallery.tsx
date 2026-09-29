import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getSpecimenGallery, type SpecimenItem } from '../../api/image';

const SPECIES_OPTIONS = ['All', 'Iris setosa', 'Iris versicolor', 'Iris virginica'] as const;

const SPECIES_COLORS: Record<string, string> = {
  'Iris setosa': '#4E79A7',
  'Iris versicolor': '#F28E2B',
  'Iris virginica': '#59A14F',
};

export default function SpecimenGallery() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialSpecies = searchParams.get('species') || 'All';

  // Normalize initial species if matching one of the options
  const matchedInitial = SPECIES_OPTIONS.find(
    s => s.toLowerCase() === initialSpecies.toLowerCase() || s.toLowerCase().includes(initialSpecies.toLowerCase())
  ) || 'All';

  const [selectedSpecies, setSelectedSpecies] = useState<string>(matchedInitial);
  const [specimens, setSpecimens] = useState<SpecimenItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [selectedSpecimen, setSelectedSpecimen] = useState<SpecimenItem | null>(null);

  useEffect(() => {
    fetchGallery(selectedSpecies);
  }, [selectedSpecies]);

  const fetchGallery = async (speciesFilter: string) => {
    setLoading(true);
    setError('');
    try {
      const filterArg = speciesFilter === 'All' ? undefined : speciesFilter;
      const res = await getSpecimenGallery(filterArg);
      setSpecimens(res.data?.items || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load botanical specimen gallery.');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (sp: string) => {
    setSelectedSpecies(sp);
    if (sp === 'All') {
      searchParams.delete('species');
      setSearchParams(searchParams);
    } else {
      setSearchParams({ species: sp });
    }
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>, item: SpecimenItem) => {
    // If external URL fails, fallback to local optimized thumbnail for the species
    const imgElement = e.currentTarget;
    const fallbackMap: Record<string, string> = {
      'Iris setosa': '/assets/iris_setosa_thumb.jpg',
      'Iris versicolor': '/assets/iris_versicolor_thumb.jpg',
      'Iris virginica': '/assets/iris_virginica_thumb.jpg',
    };
    const fallback = fallbackMap[item.species];
    if (fallback && imgElement.src !== window.location.origin + fallback) {
      imgElement.src = fallback;
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📷 Authentic Botanical Specimen Gallery</h1>
        <p className="page-subtitle">
          High-resolution authentic botanical observations, herbarium specimens, and verified field occurrences from iNaturalist Research Grade archives and Wikimedia Commons
        </p>
      </div>

      {/* Species Filter Tabs */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem' }}>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontWeight: 600, color: 'var(--text-muted)', marginRight: '8px', fontSize: '0.9rem' }}>
            Filter by Species:
          </span>
          {SPECIES_OPTIONS.map((sp) => {
            const isActive = selectedSpecies === sp;
            const color = sp === 'All' ? 'var(--primary)' : SPECIES_COLORS[sp];
            return (
              <button
                key={sp}
                onClick={() => handleTabChange(sp)}
                className={`btn-filter ${isActive ? 'active' : ''}`}
                style={{
                  padding: '6px 14px',
                  borderRadius: '20px',
                  border: `1.5px solid ${isActive ? color : 'var(--border)'}`,
                  background: isActive ? `${color}18` : 'var(--bg-card)',
                  color: isActive ? color : 'var(--text)',
                  fontWeight: isActive ? 700 : 500,
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                {sp !== 'All' && (
                  <span
                    style={{
                      display: 'inline-block',
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: color,
                      marginRight: '6px',
                    }}
                  />
                )}
                {sp}
              </button>
            );
          })}
        </div>
      </div>

      {error && <div className="alert-error" style={{ marginBottom: '1.5rem' }}>{error}</div>}

      {/* Loading Skeleton */}
      {loading && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.2rem' }}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="card" style={{ height: '360px', opacity: 0.5, animation: 'pulse 1.5s infinite' }}>
              <div style={{ height: '200px', background: 'var(--border)', borderRadius: '6px' }} />
              <div style={{ marginTop: '12px', height: '20px', width: '60%', background: 'var(--border)', borderRadius: '4px' }} />
              <div style={{ marginTop: '8px', height: '14px', width: '80%', background: 'var(--border)', borderRadius: '4px' }} />
            </div>
          ))}
        </div>
      )}

      {/* Specimen Card Grid */}
      {!loading && specimens.length > 0 && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))',
            gap: '1.4rem',
          }}
        >
          {specimens.map((specimen, idx) => {
            const spColor = SPECIES_COLORS[specimen.species] || 'var(--primary)';
            return (
              <div
                key={specimen.id || idx}
                className="card"
                style={{
                  padding: '0',
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                  border: '1px solid var(--border)',
                  borderRadius: '10px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
                  transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedSpecimen(specimen)}
              >
                {/* Specimen Photo Thumbnail */}
                <div style={{ position: 'relative', height: '210px', overflow: 'hidden', background: '#f5f5f5' }}>
                  <img
                    src={specimen.thumbnail_url || specimen.image_url}
                    alt={specimen.species}
                    loading="lazy"
                    decoding="async"
                    onError={(e) => handleImageError(e, specimen)}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                      transition: 'transform 0.3s ease',
                    }}
                  />
                  <span
                    style={{
                      position: 'absolute',
                      top: '10px',
                      left: '10px',
                      background: 'rgba(255, 255, 255, 0.92)',
                      color: spColor,
                      padding: '4px 10px',
                      borderRadius: '12px',
                      fontSize: '0.78rem',
                      fontWeight: 700,
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                    }}
                  >
                    {specimen.species}
                  </span>
                  <span
                    style={{
                      position: 'absolute',
                      bottom: '8px',
                      right: '8px',
                      background: 'rgba(31, 56, 100, 0.85)',
                      color: '#ffffff',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                    }}
                  >
                    {specimen.source}
                  </span>
                </div>

                {/* Specimen Metadata Body */}
                <div style={{ padding: '14px', flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div style={{ fontWeight: 700, fontSize: '1rem', color: spColor }}>
                    {specimen.species}
                  </div>

                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    📍 <strong>Location:</strong> {specimen.locality || specimen.country || 'Metadata unavailable'}
                  </div>

                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    🏛️ <strong>Institution:</strong> {specimen.institution || 'Metadata unavailable'}
                  </div>

                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    👤 <strong>Recorder:</strong> {specimen.recorded_by || 'Metadata unavailable'}
                  </div>

                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    📅 <strong>Observed:</strong> {specimen.event_date ? specimen.event_date.split('T')[0] : 'Metadata unavailable'}
                  </div>

                  <div style={{ marginTop: 'auto', paddingTop: '8px', borderTop: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>⚖️ {specimen.license || 'Public Domain / Open Access'}</span>
                    <span style={{ color: 'var(--primary)', fontWeight: 600, fontSize: '0.8rem' }}>Inspect 🔍</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!loading && specimens.length === 0 && !error && (
        <div className="card empty-state" style={{ marginTop: '1.5rem', textAlign: 'center', padding: '3rem' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🌸</div>
          <h3>No Specimens Found</h3>
          <p className="text-muted">No botanical specimen occurrences found for {selectedSpecies}.</p>
        </div>
      )}

      {/* Large-Image Inspection Modal */}
      {selectedSpecimen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
          }}
          onClick={() => setSelectedSpecimen(null)}
        >
          <div
            className="card"
            style={{
              maxWidth: '850px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '0',
              borderRadius: '12px',
              background: 'var(--bg-card)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <div>
                <h2 style={{ margin: 0, fontSize: '1.25rem', color: SPECIES_COLORS[selectedSpecimen.species] }}>
                  {selectedSpecimen.species}
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Specimen Record ID: <code>{selectedSpecimen.id}</code>
                </span>
              </div>
              <button
                className="btn-secondary btn-sm"
                onClick={() => setSelectedSpecimen(null)}
                style={{ fontSize: '1.1rem', cursor: 'pointer', padding: '4px 10px' }}
              >
                ✕
              </button>
            </div>

            {/* High-res Photograph */}
            <div style={{ maxHeight: '450px', overflow: 'hidden', background: '#000000', display: 'flex', justifyContent: 'center' }}>
              <img
                src={selectedSpecimen.image_url}
                alt={selectedSpecimen.species}
                onError={(e) => handleImageError(e, selectedSpecimen)}
                style={{
                  maxHeight: '450px',
                  width: 'auto',
                  maxWidth: '100%',
                  objectFit: 'contain',
                }}
              />
            </div>

            {/* Verified Specimen Metadata Sheet */}
            <div style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '1.05rem', marginBottom: '12px', color: 'var(--text)' }}>
                🌿 Botanical Herbarium & Occurrence Record
              </h3>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: '12px',
                  fontSize: '0.88rem',
                }}
              >
                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Taxon / Species</div>
                  <strong style={{ color: SPECIES_COLORS[selectedSpecimen.species] }}>{selectedSpecimen.species}</strong>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Primary Data Provider</div>
                  <strong>{selectedSpecimen.source}</strong>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Geographic Locality</div>
                  <span>{selectedSpecimen.locality || selectedSpecimen.country || 'Metadata unavailable'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Herbarium / Institution</div>
                  <span>{selectedSpecimen.institution || 'Metadata unavailable'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Collector / Recorder</div>
                  <span>{selectedSpecimen.recorded_by || 'Metadata unavailable'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Observation Date</div>
                  <span>{selectedSpecimen.event_date ? selectedSpecimen.event_date.split('T')[0] : 'Metadata unavailable'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>Intellectual Property / License</div>
                  <span>{selectedSpecimen.license || 'Public Domain / CC0'}</span>
                </div>
              </div>

              <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <a
                  href={selectedSpecimen.image_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary btn-sm"
                  style={{ textDecoration: 'none' }}
                >
                  🔗 Open Raw Image
                </a>
                <button className="btn-primary btn-sm" onClick={() => setSelectedSpecimen(null)}>
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
