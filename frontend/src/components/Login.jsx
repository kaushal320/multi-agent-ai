import { useState } from "react";

import { loginWithFirebaseToken } from "../api.js";
import {
  signInWithEmail,
  signInWithGoogle,
  signUpWithEmail,
} from "../firebase.js";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("signin");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function backendLogin(firebaseToken) {
    await loginWithFirebaseToken(firebaseToken);
  }

  async function handleGoogle() {
    setError("");
    setBusy(true);
    try {
      const token = await signInWithGoogle();
      await backendLogin(token);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleEmail(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const token =
        mode === "signin"
          ? await signInWithEmail(email, password)
          : await signUpWithEmail(email, password);
      await backendLogin(token);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">Cortex AI</div>
        <p className="login-subtitle">Sign in to test the agent</p>

        <button className="btn btn-google" onClick={handleGoogle} disabled={busy}>
          Continue with Google
        </button>

        <div className="divider"><span>or</span></div>

        <form onSubmit={handleEmail}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
          />
          <button className="btn btn-primary" disabled={busy}>
            {mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        <button
          className="link"
          onClick={() => setMode(mode === "signin" ? "signup" : "signin")}
        >
          {mode === "signin"
            ? "New here? Create an account"
            : "Already have an account? Sign in"}
        </button>

        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
