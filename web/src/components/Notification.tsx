import { toast} from "sonner"
import { CheckIcon, X } from "lucide-react"

export default function Notification({message, description, error} : {message: string, description: string, error: boolean}) {

      return () => toast(message, {
          description: description,
          action: {
            label: error ? <X className="text-red-500 w-4 h-4"/> : <CheckIcon className="text-green-400 w-4 h-4"/>,
            onClick: () => 0,
          },
        })
}