import api from './client';

export const register = (username: string, password: string, confirmPassword: string) =>
  api.post('/auth/register', { username, password, confirm_password: confirmPassword });

export const login = (username: string, password: string) =>
  api.post('/auth/login', { username, password });

export const getMe = () => api.get('/auth/me');
