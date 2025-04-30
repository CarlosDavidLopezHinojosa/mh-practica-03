"use client"

import { ReactNode } from "react"
import { Separator } from "@/components/ui/separator"

export default function AppSidebar({ children }: { children: ReactNode }) {
  return (
    <aside className="w-64 h-screen bg-gray-100 p-4 shadow-md">
      <h2 className="text-xl font-bold mb-4">Opciones</h2>
      <div className="flex flex-col gap-4">
        {Array.isArray(children) ? (
          children.map((child, index) => (
            <div key={index}>
              {child}
              {index < children.length - 1 && <Separator className="my-4" />}
            </div>
          ))
        ) : (
          children
        )}
      </div>
    </aside>
  )
}
