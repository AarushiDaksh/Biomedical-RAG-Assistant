export default function Loading() {
  return (
    <div
      className="flex min-h-screen items-center justify-center px-6"
      style={{
        background: "#040d1a",
        backgroundImage: `
          radial-gradient(circle at top, rgba(34,211,238,0.08), transparent 28%),
          linear-gradient(rgba(6,182,212,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(6,182,212,0.04) 1px, transparent 1px)
        `,
        backgroundSize: "auto, 40px 40px, 40px 40px",
      }}
    >
      <div
        className="w-full max-w-md overflow-hidden rounded-3xl border"
        style={{
          background: "linear-gradient(180deg, rgba(9,19,35,0.98) 0%, rgba(7,15,29,0.98) 100%)",
          borderColor: "rgba(34,211,238,0.16)",
          boxShadow: "0 24px 60px rgba(0,0,0,0.34)",
        }}
      >
        <div className="px-6 py-5" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
          <div className="flex items-center gap-3">
            <div
              className="relative flex h-12 w-12 items-center justify-center rounded-2xl"
              style={{ background: "linear-gradient(135deg,#0e7490,#4f46e5)" }}
            >
              <span className="absolute inset-0 rounded-2xl border border-cyan-300/25 animate-[loaderPulse_2s_ease-in-out_infinite]" />
              <span className="relative h-5 w-5 rounded-full border-2 border-white/80 border-t-transparent animate-spin" />
            </div>
            <div>
              <p className="font-display text-lg font-semibold text-white">Flamingo</p>
              <p className="text-sm text-slate-400">Preparing your research workspace</p>
            </div>
          </div>
        </div>

        <div className="space-y-4 px-6 py-5">
          <div className="grid gap-3">
            {["Connecting to indexed papers", "Warming up the chat surface", "Staging citation-aware responses"].map((label, index) => (
              <div
                key={label}
                className="flex items-center gap-3 rounded-2xl border px-4 py-3"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  borderColor: "rgba(255,255,255,0.06)",
                  animationDelay: `${index * 120}ms`,
                }}
              >
                <span className="loader-beacon h-2.5 w-2.5 rounded-full bg-cyan-300" />
                <span className="text-sm text-slate-200">{label}</span>
              </div>
            ))}
          </div>

          <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
            <div className="loader-track h-full w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
