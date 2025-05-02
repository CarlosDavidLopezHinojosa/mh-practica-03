"use client"

import { useCallback } from "react"
import { FileLineChart } from "lucide-react"


import { Button } from "./ui/button"
import Notification from "./Notification"

export async function uploadCSVFile(file: File): Promise<number[]> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch("http://localhost:8000/dataframe", {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error("Error al procesar el archivo en el servidor.")
  }

  const data = await response.json()
  return data["data"] as number[]
}


type Props = {
  fileName: string | null
  setFileName: React.Dispatch<React.SetStateAction<string | null>>
  fileInputRef: React.RefObject<HTMLInputElement | null>
  setChartData: React.Dispatch<React.SetStateAction<number[]>>
  setDatasetName: React.Dispatch<React.SetStateAction<string>>
  toUpload: boolean
  setToUpload: React.Dispatch<React.SetStateAction<boolean>>
}

export default function CSVUploader({
  fileName,
  setFileName,
  fileInputRef,
  setChartData,
  setDatasetName,
  toUpload,
  setToUpload,
}: Props) {
  const handleButtonClick = () => {
    fileInputRef.current?.click()
    setToUpload(false)
  }

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file || file.type !== "text/csv") {
        return Notification({
          message: "Error al subir el archivo",
          description: "Solo se aceptan archivos CSV.",
          error: true,
        })()
      }

      try {
        setFileName(file.name)
        const data = await uploadCSVFile(file)
        setChartData(data)
        setDatasetName(file.name)

        Notification({
          message: "Archivo subido con éxito",
          description: `El archivo "${file.name}" se ha procesado correctamente.`,
          error: false,
        })()
      } catch {
        Notification({
          message: "Error al procesar el archivo",
          description: "Hubo un problema al procesar el archivo en el servidor.",
          error: true,
        })()
      }
    },
    [setFileName, setChartData, setDatasetName]
  )

  const renderFileName = () => {
    if (!fileName || toUpload) return "Seleccionar archivo CSV"
    const displayName = fileName.length > 20 ? `${fileName.slice(0, 17)}...` : fileName
    return (
      <>
        <FileLineChart className="mr-2" />
        {displayName}
      </>
    )
  }

  return (
    <>
      <input
        type="file"
        accept=".csv"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      <Button onClick={handleButtonClick} variant="outline" className="cursor-pointer mb-4 hover:text-green-700 hover:scale-105 transition-transform duration-200 ease-in-out">
        {renderFileName()}
      </Button>

    </>
  )
}
