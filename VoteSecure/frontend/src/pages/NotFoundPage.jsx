"use client"

import React, { useState, useEffect, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { ChevronRight, Shield, Smartphone } from "lucide-react"

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

export default function OtpVerificationPage() {
  const navigate = useNavigate()
  const [otp, setOtp] = useState(["", "", "", "", "", ""])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const [maskedPhone, setMaskedPhone] = useState("")
  const [resendDisabled, setResendDisabled] = useState(true)
  const [countdown, setCountdown] = useState(30)
  const inputRefs = useRef([])
  const [matchedVoter, setMatchedVoter] = useState(null)

  // Get masked phone from session storage
  useEffect(() => {
    const storedPhone = sessionStorage.getItem("maskedPhone")
    const storedVoter = sessionStorage.getItem("matchedVoter")

    if (storedPhone) {
      setMaskedPhone(storedPhone)
    }

    if (storedVoter) {
      setMatchedVoter(JSON.parse(storedVoter))
    } else {
      // If no matched voter, redirect back to Aadhaar verification
      navigate("/verify-aadhaar")
    }

    // Start countdown for resend button
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer)
          setResendDisabled(false)
          return 0
        }
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [navigate])

  // Handle OTP input change
  const handleOtpChange = (index, value) => {
    // Only allow numbers
    if (!/^\d*$/.test(value)) return

    const newOtp = [...otp]
    newOtp[index] = value

    setOtp(newOtp)

    // Clear error when user types
    if (error) setError(null)

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1].focus()
    }
  }

  // Handle key down events for backspace
  const handleKeyDown = (index, e) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      inputRefs.current[index - 1].focus()
    }
  }

  // Handle paste event
  const handlePaste = (e) => {
    e.preventDefault()
    const pastedData = e.clipboardData.getData("text/plain").trim()

    if (!/^\d+$/.test(pastedData)) return

    const digits = pastedData.split("").slice(0, 6)
    const newOtp = [...otp]

    digits.forEach((digit, index) => {
      if (index < 6) newOtp[index] = digit
    })

    setOtp(newOtp)

    // Focus the next empty input or the last input
    const nextEmptyIndex = newOtp.findIndex((digit) => !digit)
    if (nextEmptyIndex !== -1) {
      inputRefs.current[nextEmptyIndex].focus()
    } else if (digits.length < 6) {
      inputRefs.current[digits.length].focus()
    } else {
      inputRefs.current[5].focus()
    }
  }

  // Resend OTP
  const resendOtp = () => {
    setResendDisabled(true)
    setCountdown(30)

    // Simulate OTP resend
    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer)
          setResendDisabled(false)
          return 0
        }
        return prev - 1
      })
    }, 1000)
  }

  // Verify OTP
  const verifyOtp = () => {
    const otpValue = otp.join("")

    if (otpValue.length !== 6) {
      setError("Please enter a valid 6-digit OTP")
      return
    }

    setIsSubmitting(true)

    // Simulate API call to verify OTP
    setTimeout(() => {
      setIsSubmitting(false)

      // For demo purposes, any 6-digit OTP is considered valid
      // In a real app, this would validate against an API
      if (otpValue === "123456" || otpValue.length === 6) {
        // Store user data in session storage
        const userData = {
          name: matchedVoter?.name || "John Doe",
          voterID: matchedVoter?.id || matchedVoter?.voterID || "VID12345678",
          aadhaar: sessionStorage.getItem("aadhaarNumber")?.replace(/\s/g, "") || "123456789012",
          phone: matchedVoter?.phone || "9876543210",
          email: matchedVoter?.email || "user@example.com",
          dob: matchedVoter?.dob || "01/01/1990",
          gender: matchedVoter?.gender || "Male",
          address: matchedVoter?.address || "123 Main Street, City, State - 123456",
        }

        sessionStorage.setItem("userData", JSON.stringify(userData))

        // Navigate to success page
        navigate("/success")
      } else {
        setError("Invalid OTP. Please try again.")
      }
    }, 1500)
  }

  // Go back to Aadhaar verification
  const goBack = () => {
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
              Voter Registration
            </span>
          </h1>
          <p className="mt-2 text-center text-sm text-gray-600">Step 2: OTP Verification</p>
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
                  <Smartphone className="h-5 w-5 text-white" />
                </div>
                <div className="text-right">
                  <span className="text-sm font-medium text-gray-500">Step 2 of 2</span>
                </div>
              </div>
              <CardTitle className="mt-4 text-xl font-bold">OTP Verification</CardTitle>
              <CardDescription>Enter the 6-digit OTP sent to your phone {maskedPhone}</CardDescription>
            </CardHeader>

            <CardContent className="relative px-6 pt-6">
              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-center space-x-2">
                    {otp.map((digit, index) => (
                      <Input
                        key={index}
                        ref={(el) => (inputRefs.current[index] = el)}
                        type="text"
                        inputMode="numeric"
                        maxLength={1}
                        value={digit}
                        onChange={(e) => handleOtpChange(index, e.target.value)}
                        onKeyDown={(e) => handleKeyDown(index, e)}
                        onPaste={index === 0 ? handlePaste : undefined}
                        className="h-12 w-12 rounded-lg border-gray-200 bg-white p-0 text-center text-xl font-semibold shadow-sm transition-all focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
                      />
                    ))}
                  </div>
                  {error && <p className="text-center text-xs font-medium text-red-500">{error}</p>}
                  <p className="text-center text-xs text-gray-500">
                    Enter the 6-digit OTP sent to your registered mobile number
                  </p>
                </div>

                <div className="text-center">
                  <button
                    onClick={resendOtp}
                    disabled={resendDisabled}
                    className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline disabled:text-gray-400 disabled:no-underline"
                  >
                    {resendDisabled ? `Resend OTP in ${countdown}s` : "Resend OTP"}
                  </button>
                </div>

                <div className="rounded-lg border border-blue-100 bg-blue-50 p-4">
                  <div className="flex items-start space-x-3">
                    <Shield className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-800">Verification Information</h4>
                      <p className="mt-1 text-xs text-blue-700">
                        For demo purposes, any 6-digit OTP will work. In a real application, this would validate against
                        an API.
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
                    onClick={verifyOtp}
                    disabled={isSubmitting || otp.some((digit) => !digit)}
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
                        Verify OTP
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

