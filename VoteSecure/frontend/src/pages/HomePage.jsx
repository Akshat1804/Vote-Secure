"use client"

import React from "react"
import { useNavigate } from "react-router-dom"
import { ChevronRight } from "lucide-react"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

// Button component
const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? React.Fragment : "button"

    let variantClasses = ""
    if (variant === "default") variantClasses = "bg-primary text-primary-foreground hover:bg-primary/90"
    else if (variant === "destructive")
      variantClasses = "bg-destructive text-destructive-foreground hover:bg-destructive/90"
    else if (variant === "outline")
      variantClasses = "border border-input bg-background hover:bg-accent hover:text-accent-foreground"
    else if (variant === "secondary") variantClasses = "bg-secondary text-secondary-foreground hover:bg-secondary/80"
    else if (variant === "ghost") variantClasses = "hover:bg-accent hover:text-accent-foreground"
    else if (variant === "link") variantClasses = "text-primary underline-offset-4 hover:underline"

    let sizeClasses = ""
    if (size === "default") sizeClasses = "h-10 px-4 py-2"
    else if (size === "sm") sizeClasses = "h-9 rounded-md px-3"
    else if (size === "lg") sizeClasses = "h-11 rounded-md px-8"
    else if (size === "icon") sizeClasses = "h-10 w-10"

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
Button.displayName = "Button"

// Card components
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
))
CardFooter.displayName = "CardFooter"

export default function HomePage() {
  const navigate = useNavigate()

  const startRegistration = () => {
    navigate("/verify-aadhaar")
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-6">
        <div
          style={{
            animation: "fadeIn 0.5s ease-out",
          }}
        >
          <h1 className="text-center text-3xl font-bold tracking-tight text-gray-900">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              VoteSecure
            </span>
          </h1>
          <p className="mt-2 text-center text-sm text-gray-600">Aadhaar-based Secure Voting System</p>
        </div>

        <div
          style={{
            animation: "fadeIn 0.5s ease-out",
          }}
        >
          <Card className="overflow-hidden border-none bg-white/90 shadow-2xl backdrop-blur-sm">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 opacity-50" />
            <div className="absolute -top-10 -right-10 h-40 w-40 rounded-full bg-gradient-to-br from-blue-500/10 to-purple-500/10 blur-3xl" />
            <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-gradient-to-br from-purple-500/10 to-blue-500/10 blur-3xl" />

            <CardHeader className="relative space-y-1 pb-4">
              <CardTitle className="text-center text-2xl font-bold">Welcome to VoteSecure</CardTitle>
              <CardDescription className="text-center">
                Register with your Aadhaar to complete your voter registration
              </CardDescription>
            </CardHeader>

            <CardContent className="relative space-y-6 px-6 pt-2">
              <div className="rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 p-6">
                <h4 className="mb-3 text-center text-lg font-medium text-gray-800">How it works</h4>
                <ol className="space-y-3 text-sm text-gray-700">
                  <li className="flex items-start">
                    <span className="mr-2 flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-600">
                      1
                    </span>
                    <span>Verify your identity with your Aadhaar number</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-600">
                      2
                    </span>
                    <span>Confirm with OTP sent to your Aadhaar-linked mobile</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-600">
                      3
                    </span>
                    <span>Complete your voter registration</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-xs font-bold text-blue-600">
                      4
                    </span>
                    <span>Vote securely on election day with your voter ID</span>
                  </li>
                </ol>
              </div>

              <Button
                onClick={startRegistration}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white transition-all duration-300 hover:from-blue-700 hover:to-purple-700"
              >
                <span className="flex items-center justify-center">
                  Start Registration
                  <ChevronRight className="ml-1 h-4 w-4" />
                </span>
              </Button>
            </CardContent>

            <CardFooter className="relative mt-4 flex flex-col space-y-2 border-t border-gray-100 bg-gray-50 px-6 py-4 text-center text-xs text-gray-500">
              <p>
                Already registered?{" "}
                <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">
                  Check status
                </a>
              </p>
              <p>
                Having trouble?{" "}
                <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">
                  Contact support
                </a>
              </p>
            </CardFooter>
          </Card>
        </div>

        {/* Add custom CSS for animations */}
        <style jsx>{`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}</style>
      </div>
    </div>
  )
}

