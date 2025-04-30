"use client"

import { useState, useMemo } from "react"
import { TrendingUp } from "lucide-react"
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Slider } from "@/components/ui/slider"

type LineChartPlotProps = {
  chartData: number[]
  datasetName: string
}

const chartConfig: ChartConfig = {
  desktop: {
    label: "Desktop",
    color: "hsl(var(--chart-1))",
  },
}

export default function LineChartPlot({ chartData, datasetName }: LineChartPlotProps) {
  const initialRange = useMemo(
    () => [
      Math.floor(chartData.length / 4),
      Math.floor((chartData.length / 4) * 3),
    ],
    [chartData.length]
  )

  const [range, setRange] = useState<number[]>(initialRange)

  const slicedData = useMemo(
    () =>
      chartData
        .slice(range[0], range[1])
        .map((value, index) => ({
          index: range[0] + index,
          value,
        })),
    [chartData, range]
  )

  const tickInterval = Math.ceil(slicedData.length / 10)

  return (
    <div className="w-4xl mx-auto mt-30">
      <Card>
        <CardHeader>
          <CardTitle>{`Dataset: ${datasetName}`}</CardTitle>
          <CardDescription>{`Total de puntos: ${chartData.length}`}</CardDescription>
        </CardHeader>

        <CardContent>
          <ChartContainer config={chartConfig}>
            <LineChart
              data={slicedData}
              margin={{ top: 20, left: 12, right: 12 }}
            >
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="index"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                interval={tickInterval - 1}
              />
              <YAxis hide />
              <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent hideLabel />}
              />
              <Line
                dataKey="value"
                type="natural"
                stroke="var(--color-desktop)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>

        <CardFooter className="flex-col items-start gap-4 text-sm">
          <div className="flex gap-2 font-medium leading-none">
            Visualización de la serie temporal
            <TrendingUp className="h-4 w-4" />
          </div>

          <div className="w-full">
            <Slider
              value={range}
              onValueChange={setRange}
              min={0}
              max={chartData.length}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-3">
              <span>{`Inicio: ${range[0]}`}</span>
              <span>{`Fin: ${range[1]}`}</span>
            </div>
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}
