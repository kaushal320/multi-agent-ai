import { useCallback, useEffect, useState } from "react";
import { onAuthStateChanged, signOut } from "firebase/auth";

import { firebaseConfigured, auth } from "./firebase.js";
import { getMe, loginWithFirebaseToken } from "./api.js";
import Chat from "./components/Chat.jsx";
import Login from "./components/Login.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);
  const [sessionError, setSessionError] = useState("");

  const ensureBackendSession = useCallback(async (fbUser) => {
    setSessionError("");
    try {
      await getMe();
    } catch {
      const token = await fbUser.getIdToken(true);
      await loginWithFirebaseToken(token);
    }
  }, []);

  useEffect(() => {
    if (!firebaseConfigured || !auth) {
      setChecking(false);
      return;
    }
    const unsubscribe = onAuthStateChanged(auth, async (fbUser) => {
      if (fbUser) {
        try {
          await ensureBackendSession(fbUser);
          setUser(fbUser);
        } catch (err) {
          setUser(null);
          setSessionError(err.message);
        }
      } else {
        setUser(null);
      }
      setChecking(false);
    });
    return () => unsubscribe();
  }, [ensureBackendSession]);

  async function handleSignOut() {
    await signOut(auth);
    setSessionError("");
  }

  if (!firebaseConfigured) {
    return (
      <div className="config-error">
        <h1>Firebase is not configured</h1>
        <p>
          Create <code>frontend/.env</code> from <code>.env.example</code> and fill in
          the <code>VITE_FIREBASE_*</code> values from your Firebase project
          (Project settings → Your apps → Web).
        </p>
      </div>
    );
  }

  if (checking) {
    return <div className="loading-screen">Loading…</div>;
  }

  if (sessionError && !user) {
    return (
      <div className="config-error">
        <h1>Could not connect to the backend</h1>
        <p>{sessionError}</p>
        <p className="hint">
          Make sure the FastAPI server is running on port 8000 and that your clock is
          in sync, then retry.
        </p>
        <button
          className="btn btn-primary"
          onClick={() => {
            setChecking(true);
            auth.currentUser
              ? ensureBackendSession(auth.currentUser)
                  .then(() => {
                    setUser(auth.currentUser);
                    setChecking(false);
                  })
                  .catch((err) => {
                    setSessionError(err.message);
                    setChecking(false);
                  })
              : setChecking(false);
          }}
        >
          Retry
        </button>
        <button className="btn btn-ghost" onClick={handleSignOut}>
          Sign out
        </button>
      </div>
    );
  }

  return user ? <Chat firebaseUser={user} /> : <Login />;
}
