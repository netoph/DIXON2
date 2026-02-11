import { useState, useEffect, useCallback } from 'react'
import PredictionTable from './components/PredictionTable'
import ModelStatus from './components/ModelStatus'
import ModelInfo from './components/ModelInfo'
import Header from './components/Header'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

function App() {
    const [predictions, setPredictions] = useState(null)
    const [status, setStatus] = useState(null)
    const [activeTab, setActiveTab] = useState('progol')
    const [loading, setLoading] = useState(true)
    const [refreshing, setRefreshing] = useState(false)
    const [error, setError] = useState(null)

    const fetchData = useCallback(async () => {
        try {
            setError(null)
            const [predRes, statusRes] = await Promise.all([
                fetch(`${API_BASE}/predictions`),
                fetch(`${API_BASE}/status`)
            ])

            if (predRes.ok) {
                const predData = await predRes.json()
                setPredictions(predData)
            } else {
                throw new Error(`Predictions: ${predRes.statusText}`)
            }

            if (statusRes.ok) {
                const statusData = await statusRes.json()
                setStatus(statusData)
            }
        } catch (err) {
            console.error('Fetch error:', err)
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }, [])

    useEffect(() => {
        fetchData()
        // Poll every 5 minutes
        const interval = setInterval(fetchData, 5 * 60 * 1000)
        return () => clearInterval(interval)
    }, [fetchData])

    const handleRefresh = async () => {
        setRefreshing(true)
        setError(null)
        try {
            const res = await fetch(`${API_BASE}/refresh`, { method: 'POST' })
            if (res.ok) {
                // Wait a moment, then fetch fresh data
                await new Promise(r => setTimeout(r, 1000))
                await fetchData()
            } else {
                throw new Error(`Refresh failed: ${res.statusText}`)
            }
        } catch (err) {
            console.error('Refresh error:', err)
            setError(err.message)
        } finally {
            setRefreshing(false)
        }
    }

    const currentPredictions = predictions
        ? activeTab === 'progol'
            ? predictions.progol
            : predictions.revancha
        : []

    return (
        <div className="min-h-screen pb-12">
            {/* Header */}
            <Header
                concurso={predictions?.concurso}
                status={status}
                onRefresh={handleRefresh}
                refreshing={refreshing}
            />

            {/* Main content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
                {/* Model Status Bar */}
                <ModelStatus status={status} loading={loading} />

                {/* Tabs */}
                <div className="flex items-center gap-3 mt-8 mb-6">
                    <button
                        className={`tab-btn ${activeTab === 'progol' ? 'active' : ''}`}
                        onClick={() => setActiveTab('progol')}
                    >
                        <span className="mr-2">⚽</span>
                        Progol
                        <span className="ml-2 text-xs opacity-60">14 partidos</span>
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'revancha' ? 'active' : ''}`}
                        onClick={() => setActiveTab('revancha')}
                    >
                        <span className="mr-2">🔄</span>
                        Revancha
                        <span className="ml-2 text-xs opacity-60">7 partidos</span>
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'modelo' ? 'active' : ''}`}
                        onClick={() => setActiveTab('modelo')}
                    >
                        <span className="mr-2">🧠</span>
                        Modelo
                        <span className="ml-2 text-xs opacity-60">Info</span>
                    </button>
                </div>

                {/* Error banner */}
                {error && (
                    <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 animate-fade-in">
                        <div className="flex items-center gap-3">
                            <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
                            </svg>
                            <span className="text-sm">{error}</span>
                        </div>
                    </div>
                )}

                {/* Loading state */}
                {loading ? (
                    <div className="flex flex-col items-center justify-center py-20 animate-fade-in">
                        <div className="relative">
                            <div className="w-16 h-16 rounded-full border-4 border-surface-700 border-t-brand-500 animate-spin"></div>
                            <div className="absolute inset-0 w-16 h-16 rounded-full border-4 border-transparent border-b-emerald-500/30 animate-spin" style={{ animationDuration: '1.5s', animationDirection: 'reverse' }}></div>
                        </div>
                        <p className="mt-6 text-surface-200 font-medium">Cargando predicciones...</p>
                        <p className="mt-2 text-sm text-slate-500">Ejecutando modelo Dixon-Coles</p>
                    </div>
                ) : activeTab === 'modelo' ? (
                    <ModelInfo />
                ) : (
                    /* Predictions Table */
                    <PredictionTable
                        predictions={currentPredictions}
                        type={activeTab}
                    />
                )}
            </main>

            {/* Footer */}
            <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16 pb-8">
                <div className="border-t border-white/5 pt-8 text-center">
                    <p className="text-xs text-slate-600">
                        Modelo Dixon-Coles (1997) · Predicciones para fines informativos únicamente
                    </p>
                    <p className="text-xs text-slate-700 mt-1">
                        Datos: FBref · Lotería Nacional de México
                    </p>
                </div>
            </footer>
        </div>
    )
}

export default App
