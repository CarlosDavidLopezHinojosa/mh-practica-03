import "./App.css"
import { Routes, Route } from "react-router-dom"
import DashBoard from "./components/Dashboard"
function App() {
  return (
    <Routes>
      <Route path="/" element={<DashBoard />} />
    </Routes>
  )
}

export default App
