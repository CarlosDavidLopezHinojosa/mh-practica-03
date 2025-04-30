import { useEffect, useRef, useState } from "react"
import { FileText, BookOpen, PlayCircle } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"

import LineChartPlot from "./LineChartPlot"
import CSVUploader from "./CSVUploader"
import Sidebar from "./AppSidebar"


function DashBoard() {
  const [fileName, setFileName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const Data = Array.from({ length: 100 }, () => Math.floor(Math.random() * 10))

  const [chartData, setChartData] = useState<number[]>(Data)
  const [datasets, setDatasets] = useState<{ [key: string]: number[] }>({})
  const [selectedDataset, setSelectedDataset] = useState<string>("Serie temporal de ejemplo")
  const [toUpload, setToUpload] = useState(true)

  useEffect(() => {
    fetch("http://localhost:8000/datasets")
      .then((response) => response.json())
      .then((data) => setDatasets(data))
      .catch((error) => console.error("Error fetching datasets:", error))
  }, [])

  return (
    <div className="flex">
      <Sidebar>
        {/* === DOCUMENTATION SECTION === */}
        <div className="space-y-2 mb-4">
          <Button variant="ghost" className="w-full justify-start gap-2">
            <FileText className="w-4 h-4" />
            Documentación
          </Button>
          <Button variant="ghost" className="w-full justify-start gap-2">
            <BookOpen className="w-4 h-4" />
            Manual de Código
          </Button>
        </div>

        <Separator className="bg-gray-400"/>

        {/* === DATASET SECTION === */}
        <div className="space-y-4 my-4">
          <div className="font-semibold text-sm text-muted-foreground">Selecciona un dataset</div>
          <Select
            onValueChange={(value) => {
              if (datasets[value]) {
                setChartData(datasets[value])
                setSelectedDataset(value)
                setToUpload(true)
              }
            }}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Selecciona un dataset" />
            </SelectTrigger>
            <SelectContent className="bg-gray-50">
              {Object.keys(datasets).map((datasetName) => (
                <SelectItem key={datasetName} value={datasetName}>
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

        <Separator  className="bg-gray-400"/>

        {/* === EXECUTION SECTION === */}
        <div className="mt-4">
          <Button
            variant="outline"
            className="w-full justify-start gap-2"
            onClick={() =>
              console.log("Ejecutando algoritmo con el archivo:", fileName)
            }
          >
            <PlayCircle className="w-4 h-4" />
            Ejecutar Algoritmo
          </Button>
        </div>
      </Sidebar>

      <main className="flex-1 p-4">
        <LineChartPlot chartData={chartData} datasetName={selectedDataset} />
      </main>
    </div>
  )
}

export default DashBoard
