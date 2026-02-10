function ModelStatus({ status, loading }) {
    if (loading || !status) {
        return (
            <div className="glass-card p-5 animate-pulse">
                <div className="flex gap-8">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="flex-1">
                            <div className="h-3 bg-surface-700 rounded w-20 mb-3"></div>
                            <div className="h-5 bg-surface-700 rounded w-16"></div>
                        </div>
                    ))}
                </div>
            </div>
        )
    }

    const metrics = [
        {
            label: 'Última Actualización',
            value: status.last_updated
                ? new Date(status.last_updated).toLocaleString('es-MX', {
                    dateStyle: 'short',
                    timeStyle: 'short',
                })
                : '—',
            icon: '🕐',
            color: 'text-slate-300',
        },
        {
            label: 'ρ (Rho)',
            value: status.rho != null ? status.rho.toFixed(4) : '—',
            icon: '📐',
            color: 'text-brand-400',
            mono: true,
            tooltip: 'Parámetro de correlación Dixon-Coles para marcadores bajos',
        },
        {
            label: 'ξ (Xi)',
            value: status.xi != null ? status.xi.toFixed(4) : '—',
            icon: '⏳',
            color: 'text-amber-400',
            mono: true,
            tooltip: 'Parámetro de decaimiento temporal — mayor = más peso a partidos recientes',
        },
        {
            label: 'Ventaja Local',
            value: status.home_advantage != null
                ? `+${(status.home_advantage * 100).toFixed(1)}%`
                : '—',
            icon: '🏟️',
            color: 'text-emerald-400',
        },
        {
            label: 'Partidos',
            value: status.n_matches?.toLocaleString() || '—',
            icon: '📊',
            color: 'text-slate-300',
        },
        {
            label: 'Equipos',
            value: status.n_teams || '—',
            icon: '👥',
            color: 'text-slate-300',
        },
        {
            label: 'Convergencia',
            value: status.convergence ? '✅ Sí' : '❌ No',
            icon: '',
            color: status.convergence ? 'text-emerald-400' : 'text-rose-400',
        },
        {
            label: 'Calidad de Datos',
            value: status.data_quality === 'clean'
                ? '✅ Sin NaN'
                : status.data_quality === 'error'
                    ? '⚠️ Error'
                    : '—',
            icon: '',
            color: status.data_quality === 'clean' ? 'text-emerald-400' : 'text-amber-400',
        },
    ]

    return (
        <div className="glass-card p-5 animate-fade-in">
            <div className="flex items-center gap-2 mb-4">
                <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    Estado del Modelo
                </h2>
                {status.error && (
                    <span className="text-xs bg-rose-500/10 text-rose-400 px-2 py-0.5 rounded-full border border-rose-500/20">
                        Error
                    </span>
                )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-4">
                {metrics.map((m, i) => (
                    <div key={i} className="group" title={m.tooltip || ''}>
                        <p className="text-[11px] text-slate-500 font-medium mb-1 truncate">
                            {m.icon && <span className="mr-1">{m.icon}</span>}
                            {m.label}
                        </p>
                        <p className={`text-sm font-semibold ${m.color} ${m.mono ? 'font-mono' : ''} group-hover:scale-105 transition-transform`}>
                            {m.value}
                        </p>
                    </div>
                ))}
            </div>

            {status.error && (
                <div className="mt-4 p-3 rounded-lg bg-rose-500/5 border border-rose-500/10 text-xs text-rose-400">
                    {status.error}
                </div>
            )}
        </div>
    )
}

export default ModelStatus
