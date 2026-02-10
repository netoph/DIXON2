function Header({ concurso, status, onRefresh, refreshing }) {
    const lastUpdated = status?.last_updated
        ? new Date(status.last_updated).toLocaleString('es-MX', {
            dateStyle: 'medium',
            timeStyle: 'short',
        })
        : 'Nunca'

    const statusClass = status?.error
        ? 'offline'
        : status?.last_updated
            ? 'online'
            : 'loading'

    return (
        <header className="glass border-b border-white/5 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo / Title */}
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-glow-blue">
                            <span className="text-xl">⚽</span>
                        </div>
                        <div>
                            <h1 className="text-lg font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                                Progol Dashboard
                            </h1>
                            <p className="text-[11px] text-slate-500 font-medium tracking-wide uppercase">
                                Dixon-Coles Model
                                {concurso && (
                                    <span className="ml-2 text-brand-400">· Concurso #{concurso}</span>
                                )}
                            </p>
                        </div>
                    </div>

                    {/* Right side: status + refresh */}
                    <div className="flex items-center gap-4">
                        {/* Status indicator */}
                        <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400">
                            <span className={`status-dot ${statusClass}`}></span>
                            <span>{lastUpdated}</span>
                        </div>

                        {/* Refresh button */}
                        <button
                            className="btn-refresh"
                            onClick={onRefresh}
                            disabled={refreshing}
                            title="Refrescar predicciones"
                        >
                            {refreshing ? (
                                <>
                                    <span className="spinner"></span>
                                    <span className="hidden sm:inline">Actualizando...</span>
                                </>
                            ) : (
                                <>
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    <span className="hidden sm:inline">Refrescar</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header
