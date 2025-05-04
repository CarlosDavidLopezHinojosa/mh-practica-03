import { useEffect, useRef, useState } from "react"
import { FileText, BookOpen, PlayCircle, CodeXml, Zap } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Switch } from "./ui/switch"
import { Toaster } from "sonner"

import LineChartPlot from "./LineChartPlot"
import CSVUploader from "./CSVUploader"
import Sidebar from "./AppSidebar"
import Notification from "./Notification"
import PSOParamsComponent from "./PSOParams"

function DashBoard() {
  const [fileName, setFileName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const Data = Array.from({ length: 100 }, () => Math.floor(Math.random() * 5))

  const [chartData, setChartData] = useState<number[]>(Data)
  const [datasets, setDatasets] = useState<{ [key: string]: number[] }>({})
  const [patterns, setPatterns] = useState<[number, number][] | undefined>(undefined)
  const [selectedDataset, setSelectedDataset] = useState<string>("Serie temporal de ejemplo")
  const [toUpload, setToUpload] = useState(true)

  // Estado para alternar entre modo Desarrollo y Producción
  const [isDevMode, setIsDevMode] = useState(true)
  const [isExecuting, setIsExecuting] = useState(false)

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
  const [PSOParams, setPSOParams] = useState<PSOParams>({
    maxLenght: 10,
    minLenght: 2,
    threshold: 0.5,
    swarmSize: 10,
    maxIterations: 10,
    inertiaWeight: 0.5,
    cognitiveWeight: 1.5,
    socialWeight: 1.5,
    mergeThreshold: 0.5,
  })

  useEffect(() => {


    // const cacheData = sessionStorage.getItem("datasets")
    // if (cacheData){
    //   setDatasets(JSON.parse(cacheData))
    //   return
    // } 
    fetch("http://localhost:8000/datasets")
      .then((response) => response.json())
      .then((data) => {
        setDatasets(data)
        // sessionStorage.setItem("datasets", JSON.stringify(datasets))
      })
      .catch((error) => {
        console.error("Error fetching datasets:", error);
        Notification({message: "Error al cargar los datasets", description: "No se pudieron cargar los datasets desde el servidor.", error: true})()
      })
  }, [])

  async function handleExecuteAlgorithm() {

    if (isExecuting) {
      Notification({
        message: "Algoritmo en ejecución",
        description: "El algoritmo ya está en ejecución. Por favor, espera.",
        error: true,
      })()
      return
    }
    
    try {
      
      setIsExecuting(true)
      Notification({
        message: "Ejecutando algoritmo",
        description: "Por favor, espera mientras se ejecuta el algoritmo.",
        error: false,
      })()
      setPatterns([])
      const response = await fetch("http://localhost:8000/pattern", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          temporal_series: chartData,
          max_lenght: PSOParams.maxLenght,
          min_lenght: PSOParams.minLenght,
          threshold: PSOParams.threshold,
          swarm_size: PSOParams.swarmSize,
          iterations: PSOParams.maxIterations,
          omega: PSOParams.inertiaWeight,
          c1: PSOParams.cognitiveWeight,
          c2: PSOParams.socialWeight,
          merge_thresh: PSOParams.mergeThreshold,
        }),
      });

      if (!response.ok) {
        Notification({
          message: "Error al ejecutar el algoritmo",
          description: "No se pudo ejecutar el algoritmo en el servidor.",
          error: true,
        })();
        return;
      }

      const data = await response.json();

      if (data.best_pattern && Array.isArray(data.best_pattern)) {
        if (data.best_pattern.length === 0) {
          Notification({
            message: "No se encontraron patrones",
            description: "El algoritmo no encontró patrones en la serie temporal.",
            error: true,
          })();
        }
        else {
        setPatterns(data.best_pattern as [number, number][]);
        Notification({
          message: "Algoritmo ejecutado",
          description: "El algoritmo se ha ejecutado correctamente.",
          error: false,
        })();
      }
      } else {
        Notification({
          message: "Error en la respuesta del servidor",
          description: "El servidor no devolvió un patrón válido.",
          error: true,
        })();
      }
    } catch (error) {
      Notification({
        message: "Error inesperado",
        description: "Ocurrió un error durante la ejecución del algoritmo.",
        error: true,
      })();
      console.error("Error executing algorithm:", error);
    }

    finally {
      setIsExecuting(false)
    }
  }

  return (
    <div className="flex">
      <Sidebar>
        {/* === DATASET SECTION === */}
        <div className="space-y-4 my-4">
          <div className="font-semibold text-sm text-muted-foreground">Selecciona un dataset</div>
          <Select
            onValueChange={(value) => {
              if (datasets[value]) {
                setChartData(datasets[value])
                setSelectedDataset(value)
                setToUpload(true)
                setPatterns([])
              }
            }}
          >
            <SelectTrigger className="cursor-pointer bg-white">
              <SelectValue placeholder="Selecciona un dataset" />
            </SelectTrigger>
            <SelectContent className="bg-gray-50">
              {Object.keys(datasets).map((datasetName) => (
                <SelectItem key={datasetName} value={datasetName} className="cursor-pointer hover:bg-gray-200">
                  {datasetName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="font-semibold text-sm text-muted-foreground">O sube el tuyo</div>
          <CSVUploader
            fileName={fileName}
            setFileName={setFileName}
            fileInputRef={fileInputRef}
            setChartData={setChartData}
            setDatasetName={setSelectedDataset}
            toUpload={toUpload}
            setToUpload={setToUpload}
          />
        </div>
        <Separator className="bg-gray-400" />

        {/* === EXECUTION SECTION === */}
        <div className="mt-4 space-y-2">


          <div className="flex items-center justify-between px-2">
            <span className={`flex items-center gap-1 text-sm font-medium ${isDevMode ? "text-green-600" : "text-orange-600"}`}>
              {isDevMode ? <CodeXml className="w-4 h-4" /> : <Zap className="w-4 h-4" />}
              Modo {isDevMode ? "Desarrollo" : "Producción"}
            </span>
            <Switch
              checked={isDevMode}
              onCheckedChange={setIsDevMode}
              className={`cursor-pointer transition-colors duration-200 ring-1 ring-black ${
                isDevMode ? "bg-green-500" : "bg-orange-500"
              }`}
            />
          </div>

          {isDevMode && <PSOParamsComponent PSOParams={PSOParams} setPSOParams={setPSOParams} />}

          <Button
            variant="outline"
            className={`${isExecuting ? "cursor-wait" : "cursor-pointer"} w-44 justify-start gap-2 transition-transform duration-200 ease-in-out 
              hover:transform hover:scale-105 ${
                isDevMode ? "hover:text-green-600" : "hover:text-orange-600"
              }`}
            onClick={() =>
              handleExecuteAlgorithm()
            }
          >
            <PlayCircle className="w-4 h-4" />
            Ejecutar Algoritmo
          </Button>

        </div>

        <Separator className="bg-gray-400" />

        {/* === DOCUMENTATION SECTION === */}
        <div className="space-y-2 mb-4">
          <Button variant="ghost" className="cursor-pointer w-full justify-start gap-2 hover:text-blue-600 hover:tranform hover:scale-105 transition-transform duration-200 ease-in-out">
            <FileText className="w-4 h-4" />
            Documentación
          </Button>
          <Button variant="ghost" className="cursor-pointer w-full justify-start gap-2 hover:text-emerald-600 hover:tranform hover:scale-105 transition-transform duration-200 ease-in-out">
            <BookOpen className="w-4 h-4" />
            Manual de Código
          </Button>
        </div>
      </Sidebar>

      <main className="flex-1 p-4">
        <LineChartPlot chartData={chartData} datasetName={selectedDataset} isDevMode={isDevMode} patterns={patterns} />
      </main>
      <Toaster richColors closeButton={false} position="top-center" />

    </div>
  )
}

export default DashBoard
