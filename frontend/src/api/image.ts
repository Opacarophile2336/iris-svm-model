import api from './client';

export interface SpecimenItem {
  id: string;
  species: string;
  image_url: string;
  thumbnail_url?: string | null;
  source: string;
  recorded_by?: string | null;
  country?: string | null;
  locality?: string | null;
  event_date?: string | null;
  license?: string | null;
  institution?: string | null;
  occurrence_url?: string | null;
}

export interface GalleryResponse {
  species: string;
  items: SpecimenItem[];
  count: number;
}

export interface ShowcaseResponse {
  species: string;
  showcase: SpecimenItem[];
  displayed_ids: string[];
  count: number;
  pool_size: number;
}

export const getSpeciesImage = (speciesName: string, exclude?: string) =>
  api.get(`/image/species/${encodeURIComponent(speciesName)}`, {
    params: exclude ? { exclude } : undefined,
  });

export const getSpecimenShowcase = (speciesName: string, excludeIds?: string[], count: number = 3) =>
  api.get<ShowcaseResponse>(`/image/showcase/${encodeURIComponent(speciesName)}`, {
    params: {
      count,
      exclude: excludeIds && excludeIds.length > 0 ? excludeIds.join(',') : undefined,
    },
  });

export const getSpecimenGallery = (species?: string) =>
  api.get<GalleryResponse>('/image/gallery', {
    params: species ? { species } : undefined,
  });

export const getSpecimenGalleryBySpecies = (speciesName: string) =>
  api.get<GalleryResponse>(`/image/gallery/${encodeURIComponent(speciesName)}`);
