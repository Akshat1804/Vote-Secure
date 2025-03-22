"use client"

import React from "react"
import { cn } from "../../utils/cn"

// Button component
export const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? "button" : "button"

    let variantClasses = ""
    if (variant === "default") variantClasses = "bg-primary text-primary-foreground hover:bg-primary/90"
    else if (variant === "destructive")
      variantClasses = "bg-destructive text-destructive-foreground hover:bg-destructive/90"
    else if (variant === "outline")
      variantClasses = "border border-input bg-background hover:bg-accent hover:text-accent-foreground"
    else if (variant === "secondary") variantClasses = "bg-secondary text-secondary-foreground hover:bg-secondary/80"
    else if (variant === "ghost") variantClasses = "hover:bg-accent hover:text-accent-foreground"
    else if (variant === "link") variantClasses = "text-primary underline-offset-4 hover:underline"
    else if (variant === "indigo") variantClasses = "bg-indigo-600 text-white hover:bg-indigo-700"
    else if (variant === "purple") variantClasses = "bg-purple-600 text-white hover:bg-purple-700"

    let sizeClasses = ""
    if (size === "default") sizeClasses = "h-10 px-4 py-2"
    else if (size === "sm") sizeClasses = "h-9 rounded-md px-3"
    else if (size === "lg") sizeClasses = "h-11 rounded-md px-8"
    else if (size === "icon") sizeClasses = "h-10 w-10"

    if (asChild) {
      return <React.Fragment ref={ref} {...props} />
    }

    return (
      <Comp
        className={cn(
          "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          variantClasses,
          sizeClasses,
          className,
        )}
        ref={ref}
        {...props}
      />
    )
  },
)

// Card components
export const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
))

export const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
))

export const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
))

export const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
))

export const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))

export const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
))

// Badge component
export const Badge = React.forwardRef(({ className, variant = "default", ...props }, ref) => {
  const variantClasses = {
    default: "bg-primary/10 text-primary hover:bg-primary/20",
    secondary: "bg-secondary/10 text-secondary hover:bg-secondary/20",
    outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
    success: "bg-green-100 text-green-800 hover:bg-green-200",
    warning: "bg-yellow-100 text-yellow-800 hover:bg-yellow-200",
    danger: "bg-red-100 text-red-800 hover:bg-red-200",
    info: "bg-blue-100 text-blue-800 hover:bg-blue-200",
    purple: "bg-purple-100 text-purple-800 hover:bg-purple-200",
    indigo: "bg-indigo-100 text-indigo-800 hover:bg-indigo-200",
  }

  return (
    <span
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
        variantClasses[variant] || variantClasses.default,
        className,
      )}
      {...props}
    />
  )
})

// Tooltip component
export const Tooltip = ({ children, content, position = "top" }) => {
  const [isVisible, setIsVisible] = React.useState(false)
  const positionClasses = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  }

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          className={cn(
            "absolute z-50 w-max max-w-xs rounded-md bg-black px-3 py-1.5 text-xs text-white animate-in fade-in-50 data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1",
            positionClasses[position],
          )}
        >
          {content}
          <div
            className={cn(
              "absolute h-2 w-2 rotate-45 bg-black",
              position === "top" && "top-full left-1/2 -translate-x-1/2 -translate-y-1/2",
              position === "bottom" && "bottom-full left-1/2 -translate-x-1/2 translate-y-1/2",
              position === "left" && "left-full top-1/2 -translate-x-1/2 -translate-y-1/2",
              position === "right" && "right-full top-1/2 translate-x-1/2 -translate-y-1/2",
            )}
          />
        </div>
      )}
    </div>
  )
}

// Modal component
export const Modal = ({ isOpen, onClose, title, children }) => {
  const modalRef = React.useRef(null)

  React.useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === "Escape") onClose()
    }

    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener("keydown", handleEscape)
      document.addEventListener("mousedown", handleClickOutside)
      document.body.style.overflow = "hidden"
    }

    return () => {
      document.removeEventListener("keydown", handleEscape)
      document.removeEventListener("mousedown", handleClickOutside)
      document.body.style.overflow = "auto"
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-all">
      <div
        ref={modalRef}
        className="w-full max-w-md rounded-lg bg-white p-6 shadow-lg animate-in fade-in-50 slide-in-from-bottom-10 duration-300"
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-medium">{title}</h3>
          <button
            onClick={onClose}
            className="rounded-full p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
            aria-label="Close"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        <div>{children}</div>
      </div>
    </div>
  )
}

// Progress bar component
export const ProgressBar = ({ value, max, className }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-gray-200", className)}>
      <div
        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500 ease-in-out"
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}

