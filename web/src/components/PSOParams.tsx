import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card"

import { ChartScatter } from "lucide-react"

type PSOParams = {
  maxLenght: number
  minLenght: number
  threshold: number
  swarmSize: number
  maxIterations: number
  inertiaWeight: number
  cognitiveWeight: number
  socialWeight: number
  mergeThreshold: number
}

type PSOParamsProps = {
  PSOParams: PSOParams
  setPSOParams: (params: PSOParams) => void
}

const PARAM_LABELS: Record<keyof PSOParams, string> = {
  maxLenght: "Longitud Máx.",
  minLenght: "Longitud Mín.",
  threshold: "Umbral",
  swarmSize: "Tamaño Enjambre",
  maxIterations: "Iteraciones Máx.",
  inertiaWeight: "Inercia (w)",
  cognitiveWeight: "Cognitivo (c1)",
  socialWeight: "Social (c2)",
  mergeThreshold: "Umbral de Fusión",
}

const PARAM_DESCRIPTIONS: Record<keyof PSOParams, string> = {
  maxLenght: "Longitud máxima permitida para los patrones.",
  minLenght: "Longitud mínima permitida para los patrones.",
  threshold: "Valor de aceptación de patrones.",
  swarmSize: "Cantidad de partículas en el enjambre.",
  maxIterations: "Cantidad máxima de iteraciones del algoritmo.",
  inertiaWeight: "Influencia de la velocidad anterior en la actual (w).",
  cognitiveWeight: "Influencia del mejor valor personal encontrado (c1).",
  socialWeight: "Influencia del mejor valor global encontrado (c2).",
  mergeThreshold: "Umbral de fusión para patrones similares.",
}

const PARAM_LIMITS: Partial<Record<keyof PSOParams, { min: number; max: number }>> = {
  maxLenght: { min: 1, max: 100 },
  minLenght: { min: 1, max: 100 },
  threshold: { min: 0, max: 1 },
  swarmSize: { min: 1, max: 100 },
  maxIterations: { min: 1, max: 100 },
  inertiaWeight: { min: 0, max: 1 },
  cognitiveWeight: { min: 0, max: 5 },
  socialWeight: { min: 0, max: 5 },
  mergeThreshold: { min: 0, max: 10 },
}

export default function PSOParamsComponent({ PSOParams, setPSOParams }: PSOParamsProps) {
  const handleChange = (key: keyof PSOParams) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value
    const value = parseFloat(rawValue)

    if (isNaN(value)) return

    const limits = PARAM_LIMITS[key]
    if (limits && (value < limits.min || value > limits.max)) return

    setPSOParams({
      ...PSOParams,
      [key]: value,
    })
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" className="cursor-pointer w-44">
          <ChartScatter className="mr-2" />
          Configurar PSO
        </Button>
      </PopoverTrigger>
      <PopoverContent side="right" className="w-96 bg-gray-50">
        <div className="grid gap-4">
          <div className="space-y-2">
            <h4 className="font-medium leading-none">Parámetros del algoritmo PSO</h4>
            <p className="text-sm text-muted-foreground">
              Ajusta los valores para optimizar el comportamiento del algoritmo.
            </p>
          </div>
          <div className="grid gap-3">
            {Object.entries(PSOParams).map(([key, value]) => {
              const k = key as keyof PSOParams
              const limits = PARAM_LIMITS[k]

              return (
                <div className="grid grid-cols-3 items-center gap-4" key={k}>
                  <HoverCard>
                    <HoverCardTrigger asChild>
                      <Label htmlFor={k} className="cursor-help">
                        {PARAM_LABELS[k]}
                      </Label>
                    </HoverCardTrigger>
                    <HoverCardContent className="w-64 text-sm text-muted-foreground bg-gray-50">
                      {PARAM_DESCRIPTIONS[k]}
                    </HoverCardContent>
                  </HoverCard>

                  <Input
                    id={k}
                    type="number"
                    value={value}
                    onChange={handleChange(k)}
                    className="col-span-2 h-8"
                    {...(limits && { min: limits.min, max: limits.max })}
                  />
                </div>
              )
            })}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
