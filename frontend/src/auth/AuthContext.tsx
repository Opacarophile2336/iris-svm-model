import React, { createContext, useContext, useState, useEffect } from 'react';

export interface User {
  username: string;
  role: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (token: string, role: string, username: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  login: () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    const savedToken = localStorage.getItem('hmnc_token');
    const savedUser = localStorage.getItem('hmnc_user');
    if (savedToken && savedUser) {
      try {
        setToken(savedToken);
        setUser(JSON.parse(savedUser));
      } catch {
        localStorage.removeItem('hmnc_token');
        localStorage.removeItem('hmnc_user');
      }
    }
  }, []);

  const login = (newToken: string, role: string, username: string) => {
    const userObj: User = { username, role };
    localStorage.setItem('hmnc_token', newToken);
    localStorage.setItem('hmnc_user', JSON.stringify(userObj));
    setToken(newToken);
    setUser(userObj);
  };

  const logout = () => {
    localStorage.removeItem('hmnc_token');
    localStorage.removeItem('hmnc_user');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
