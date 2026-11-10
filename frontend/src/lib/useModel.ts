import { useAsync } from '../components/ui'
import { api } from './api'

/* eslint-disable @typescript-eslint/no-explicit-any */
let cache: Promise<any> | null = null
export const fetchModelInfo = () => (cache ??= api.modelInfo().catch((e) => { cache = null; throw e }))
export const useModelInfo = () => useAsync<any>(fetchModelInfo, [])
