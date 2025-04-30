
const env = import.meta.env 
const config = {
  apiUrl: env.VITE_API_SERVER,
  apiKey: env.VITE_API_PORT
}

console.log(config)