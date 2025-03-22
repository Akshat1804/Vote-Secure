"use client"

import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import { ChevronRight, Shield, User, AlertCircle } from "lucide-react"
import { useVoters } from "../context/VoterContext"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

// UI Components
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

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = "Input"

const Label = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <label
      className={cn(
        "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70",
        className,
      )}
      ref={ref}
      {...props}
    />
  )
})
Label.displayName = "Label"

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

export default function AadhaarVerificationPage() {
  const navigate = useNavigate()
  const { voters } = useVoters()
  const [aadhaarNumber, setAadhaarNumber] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errors, setErrors] = useState({})
  const [verificationError, setVerificationError] = useState(null)

  // Format Aadhaar number with spaces
  const formatAadhaar = (value) => {
    const digits = value.replace(/\D/g, "")
    let formatted = ""

    for (let i = 0; i < digits.length && i < 12; i++) {
      if (i > 0 && i % 4 === 0) {
        formatted += " "
      }
      formatted += digits[i]
    }

    return formatted
  }

  // Handle Aadhaar input change
  const handleAadhaarChange = (e) => {
    const formatted = formatAadhaar(e.target.value)
    setAadhaarNumber(formatted)

    // Clear errors when user types
    if (errors.aadhaar) {
      setErrors({ ...errors, aadhaar: null })
    }

    // Clear verification error when user types
    if (verificationError) {
      setVerificationError(null)
    }
  }

  // Validate Aadhaar number
  const validateAadhaar = () => {
    const aadhaarDigits = aadhaarNumber.replace(/\s/g, "")

    if (!aadhaarDigits) {
      setErrors({ ...errors, aadhaar: "Aadhaar number is required" })
      return false
    }

    if (aadhaarDigits.length !== 12) {
      setErrors({ ...errors, aadhaar: "Aadhaar number must be 12 digits" })
      return false
    }

    return true
  }

  // Submit Aadhaar for verification
  const submitAadhaar = () => {
    if (validateAadhaar()) {
      setIsSubmitting(true)
      setVerificationError(null)

      // Check if the Aadhaar exists in the voter database
      const cleanAadhaar = aadhaarNumber.replace(/\s/g, "")
      // Improved matching logic to handle different Aadhaar formats and property names
      const matchingVoter = voters.find((voter) => {
        // Check if voter has aadhaar property and it matches (after cleaning spaces)
        if (voter.aadhaar) {
          const voterAadhaar = voter.aadhaar.replace(/\s/g, "")
          return voterAadhaar === cleanAadhaar
        }
        return false
      })

      // For debugging - log to console
      console.log("Searching for Aadhaar:", cleanAadhaar)
      console.log("Available voters:", voters)
      console.log("Found matching voter:", matchingVoter)

      setTimeout(() => {
        setIsSubmitting(false)

        if (matchingVoter) {
          // Store Aadhaar in session storage to pass between pages
          sessionStorage.setItem("aadhaarNumber", aadhaarNumber)
          sessionStorage.setItem("matchedVoter", JSON.stringify(matchingVoter))

          // Generate masked phone number (in real app, this would come from API)
          const lastFourDigits = matchingVoter.phone.slice(-4)
          sessionStorage.setItem("maskedPhone", `******${lastFourDigits}`)

          // Navigate to OTP verification page
          navigate("/verify-otp")
        } else {
          // Show error message if Aadhaar is not found in the database
          setVerificationError(
            "No voter record found with this Aadhaar number. Please contact your local election office.",
          )
        }
      }, 1500)
    }
  }

  // Go back to home
  const goBack = () => {
    navigate("/")
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
              Voter Registration
            </span>
          </h1>
          <p className="mt-2 text-center text-sm text-gray-600">Step 1: Aadhaar Verification</p>
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

            <CardHeader className="relative space-y-1 border-b border-gray-100 pb-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
                  <User className="h-5 w-5 text-white" />
                </div>
                <div className="text-right">
                  <span className="text-sm font-medium text-gray-500">Step 1 of 2</span>
                </div>
              </div>
              <CardTitle className="mt-4 text-xl font-bold">Aadhaar Verification</CardTitle>
              <CardDescription>Enter your 12-digit Aadhaar number</CardDescription>
            </CardHeader>
            <div className="px-6 pt-2">
              <details className="mb-4 rounded-lg border border-gray-200 bg-gray-50 p-2 text-xs">
                <summary className="cursor-pointer font-medium text-gray-700">Debug Information</summary>
                <div className="mt-2 space-y-1 text-gray-600">
                  <p>Total voters in database: {voters.length}</p>
                  <p>Looking for Aadhaar: {aadhaarNumber}</p>
                  <div>
                    <p>Available Aadhaar numbers:</p>
                    <ul className="ml-4 list-disc">
                      {voters.map((voter, index) => (
                        <li key={index}>
                          {voter.aadhaar || "No Aadhaar"} - {voter.name}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </details>
            </div>

            <CardContent className="relative px-6 pt-6">
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="aadhaar" className="flex items-center gap-2 text-sm font-medium text-gray-700">
                    <User className="h-4 w-4 text-blue-600" />
                    Aadhaar Number
                  </Label>
                  <div className="relative">
                    <Input
                      id="aadhaar"
                      placeholder="XXXX XXXX XXXX"
                      value={aadhaarNumber}
                      onChange={handleAadhaarChange}
                      maxLength={14} // 12 digits + 2 spaces
                      className={`h-11 rounded-lg border-gray-200 bg-white pl-3 pr-3 text-center text-lg tracking-wider shadow-sm transition-all focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 ${errors.aadhaar ? "border-red-300 focus:border-red-500 focus:ring-red-200" : ""}`}
                    />
                  </div>
                  {errors.aadhaar && <p className="text-xs font-medium text-red-500">{errors.aadhaar}</p>}
                  <p className="text-xs text-gray-500">Enter your 12-digit Aadhaar number to verify your identity</p>
                </div>

                {verificationError && (
                  <div className="rounded-lg border border-red-200 bg-red-50 p-4">
                    <div className="flex items-start space-x-3">
                      <AlertCircle className="mt-0.5 h-5 w-5 text-red-600" />
                      <div>
                        <h4 className="text-sm font-medium text-red-800">Verification Failed</h4>
                        <p className="mt-1 text-xs text-red-700">{verificationError}</p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
                  <div className="flex items-start space-x-3">
                    <Shield className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-800">Why we need your Aadhaar</h4>
                      <p className="mt-1 text-xs text-blue-700">
                        Your Aadhaar is used to verify your identity and match with your voter registration details. We
                        do not store your Aadhaar details.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-4">
                  <Button
                    onClick={goBack}
                    variant="outline"
                    className="flex w-1/3 items-center border-gray-200 bg-white text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                  >
                    Back
                  </Button>

                  <Button
                    onClick={submitAadhaar}
                    disabled={isSubmitting}
                    className="relative w-2/3 overflow-hidden bg-gradient-to-r from-blue-600 to-purple-600 text-white transition-all duration-300 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400"
                  >
                    {isSubmitting ? (
                      <span className="flex items-center justify-center">
                        <svg className="mr-2 h-4 w-4 animate-spin" viewBox="0 0 24 24">
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                        Verifying...
                      </span>
                    ) : (
                      <span className="flex items-center justify-center">
                        Verify Aadhaar
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </span>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>

            <CardFooter className="relative mt-4 flex flex-col space-y-2 border-t border-gray-100 bg-gray-50 px-6 py-4 text-center text-xs text-gray-500">
              <p className="flex items-center justify-center">
                <Shield className="mr-1 h-3 w-3 text-gray-400" />
                Your data is encrypted and securely processed
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

