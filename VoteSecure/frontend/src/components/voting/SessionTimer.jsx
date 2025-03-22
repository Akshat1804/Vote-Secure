"use client"

import { useState, useEffect } from "react"
import { cn } from "../../utils/cn"

// Session timer component
const SessionTimer = ({ duration = 300, onExpire }) => {
  const [timeLeft, setTimeLeft] = useState(duration)

  useEffect(() => {
    if (timeLeft <= 0) {
      onExpire?.()
      return
    }

    const timer = setTimeout(() => {
      setTimeLeft((prev) => prev - 1)
    }, 1000)

    return () => clearTimeout(timer)
  }, [timeLeft, onExpire])

  const minutes = Math.floor(timeLeft / 60)
  const seconds = timeLeft % 60

  return (
    <div className="flex items-center space-x-1.5 text-sm">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="12 6 12 12 16 14"></polyline>
      </svg>
      <span
        className={cn(
          "font-medium",
          timeLeft < 60 ? "text-red-500" : timeLeft < 120 ? "text-yellow-500" : "text-gray-500",
        )}
      >
        {minutes.toString().padStart(2, "0")}:{seconds.toString().padStart(2, "0")}
      </span>
    </div>
  )
}

export default SessionTimer

