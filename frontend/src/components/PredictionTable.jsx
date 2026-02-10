function PredictionTable({ predictions, type }) {
    if (!predictions || predictions.length === 0) {
        return (
            <div className="glass-card p-12 text-center animate-fade-in">
                <div className="text-4xl mb-4">📋</div>
                <p className="text-slate-400 font-medium">No hay predicciones disponibles</p>
                <p className="text-xs text-slate-600 mt-2">
                    Presiona "Refrescar" para cargar las predicciones del concurso actual
                </p>
            </div>
        )
    }

    return (
        <div className="space-y-3 animate-fade-in">
            {/* Table header (desktop) */}
            <div className="hidden lg:grid grid-cols-12 gap-4 px-6 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                <div className="col-span-1">#</div>
                <div className="col-span-2">Local</div>
                <div className="col-span-2">Visitante</div>
                <div className="col-span-1 text-center">P(L)</div>
                <div className="col-span-1 text-center">P(E)</div>
                <div className="col-span-1 text-center">P(V)</div>
                <div className="col-span-1 text-center">Pick</div>
                <div className="col-span-3">Top Marcadores</div>
            </div>

            {/* Match rows */}
            {predictions.map((match, index) => (
                <MatchRow key={`${type}-${match.match_num}`} match={match} index={index} />
            ))}

            {/* Summary */}
            <SummaryBar predictions={predictions} type={type} />
        </div>
    )
}

function MatchRow({ match, index }) {
    const maxProb = Math.max(match.home_win, match.draw, match.away_win)

    return (
        <div className="match-row glass-card p-4 lg:p-5">
            <div className="lg:grid lg:grid-cols-12 lg:gap-4 lg:items-center">
                {/* Match number */}
                <div className="hidden lg:flex col-span-1 items-center">
                    <span className="w-8 h-8 rounded-lg bg-surface-700/50 flex items-center justify-center text-sm font-bold text-slate-400">
                        {match.match_num}
                    </span>
                </div>

                {/* Teams (mobile: row, desktop: columns) */}
                <div className="flex items-center justify-between lg:contents mb-3 lg:mb-0">
                    {/* Home team */}
                    <div className="col-span-2 flex items-center gap-2">
                        <span className="lg:hidden w-6 h-6 rounded bg-surface-700/50 flex items-center justify-center text-[10px] font-bold text-slate-500">
                            {match.match_num}
                        </span>
                        <div>
                            <p className="font-semibold text-sm text-white capitalize">{match.home}</p>
                            {match.league && (
                                <p className="text-[10px] text-slate-500 font-medium">{match.league}</p>
                            )}
                        </div>
                    </div>

                    {/* VS badge (mobile) */}
                    <span className="lg:hidden text-[10px] font-bold text-slate-600 px-2">VS</span>

                    {/* Away team */}
                    <div className="col-span-2 text-right lg:text-left">
                        <p className="font-semibold text-sm text-white capitalize">{match.away}</p>
                    </div>
                </div>

                {/* Probabilities */}
                <div className="grid grid-cols-3 gap-2 lg:contents my-3 lg:my-0">
                    <ProbCell
                        label="L"
                        value={match.home_win}
                        isMax={match.pick === 'L'}
                        color="emerald"
                    />
                    <ProbCell
                        label="E"
                        value={match.draw}
                        isMax={match.pick === 'E'}
                        color="amber"
                    />
                    <ProbCell
                        label="V"
                        value={match.away_win}
                        isMax={match.pick === 'V'}
                        color="rose"
                    />
                </div>

                {/* Pick */}
                <div className="hidden lg:flex col-span-1 justify-center">
                    <span className={`pick-badge pick-${match.pick}`}>
                        {match.pick}
                    </span>
                </div>

                {/* Top scorelines */}
                <div className="col-span-3 mt-3 lg:mt-0">
                    <div className="flex flex-wrap gap-1.5">
                        {match.top_scorelines?.map((score, i) => (
                            <span key={i} className="scoreline-chip">
                                <span className="text-white font-bold">{score.home_goals}</span>
                                <span className="text-slate-600">-</span>
                                <span className="text-white font-bold">{score.away_goals}</span>
                                <span className="text-[10px] text-slate-500 ml-1">
                                    {(score.probability * 100).toFixed(0)}%
                                </span>
                            </span>
                        ))}
                    </div>
                </div>

                {/* Mobile pick */}
                <div className="flex lg:hidden items-center justify-between mt-3 pt-3 border-t border-white/5">
                    <span className="text-[11px] text-slate-500 font-medium">Recomendación</span>
                    <span className={`pick-badge pick-${match.pick}`}>
                        {match.pick}
                    </span>
                </div>
            </div>
        </div>
    )
}

function ProbCell({ label, value, isMax, color }) {
    const pct = (value * 100).toFixed(1)
    const barWidth = `${Math.min(value * 100, 100)}%`

    const colorMap = {
        emerald: {
            text: isMax ? 'text-emerald-400' : 'text-slate-400',
            bar: 'bg-gradient-to-r from-emerald-500 to-emerald-400',
            barBg: 'bg-emerald-500/10',
        },
        amber: {
            text: isMax ? 'text-amber-400' : 'text-slate-400',
            bar: 'bg-gradient-to-r from-amber-500 to-amber-400',
            barBg: 'bg-amber-500/10',
        },
        rose: {
            text: isMax ? 'text-rose-400' : 'text-slate-400',
            bar: 'bg-gradient-to-r from-rose-500 to-rose-400',
            barBg: 'bg-rose-500/10',
        },
    }

    const c = colorMap[color]

    return (
        <div className="col-span-1 text-center">
            {/* Mobile label */}
            <p className="lg:hidden text-[10px] text-slate-600 font-semibold mb-1">{label}</p>

            {/* Percentage */}
            <p className={`text-sm font-bold font-mono ${c.text} transition-colors`}>
                {pct}%
            </p>

            {/* Bar */}
            <div className={`prob-bar mt-1.5 ${c.barBg}`}>
                <div
                    className={`h-full rounded-full ${c.bar} ${isMax ? 'shadow-lg' : ''}`}
                    style={{ width: barWidth }}
                />
            </div>
        </div>
    )
}

function SummaryBar({ predictions, type }) {
    const pickCounts = { L: 0, E: 0, V: 0 }
    predictions.forEach(p => pickCounts[p.pick]++)

    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length

    return (
        <div className="glass-card p-4 mt-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-6">
                    <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                        Resumen {type === 'progol' ? 'Progol' : 'Revancha'}
                    </h3>
                    <div className="flex items-center gap-4 text-sm">
                        <span className="flex items-center gap-1.5">
                            <span className="w-3 h-3 rounded-sm bg-emerald-500/20 border border-emerald-500/30"></span>
                            <span className="text-emerald-400 font-semibold">{pickCounts.L}</span>
                            <span className="text-slate-500 text-xs">Locales</span>
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="w-3 h-3 rounded-sm bg-amber-500/20 border border-amber-500/30"></span>
                            <span className="text-amber-400 font-semibold">{pickCounts.E}</span>
                            <span className="text-slate-500 text-xs">Empates</span>
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="w-3 h-3 rounded-sm bg-rose-500/20 border border-rose-500/30"></span>
                            <span className="text-rose-400 font-semibold">{pickCounts.V}</span>
                            <span className="text-slate-500 text-xs">Visitantes</span>
                        </span>
                    </div>
                </div>
                <div className="text-xs text-slate-500">
                    Confianza promedio:{' '}
                    <span className="font-mono font-semibold text-brand-400">
                        {(avgConfidence * 100).toFixed(1)}%
                    </span>
                </div>
            </div>
        </div>
    )
}

export default PredictionTable
