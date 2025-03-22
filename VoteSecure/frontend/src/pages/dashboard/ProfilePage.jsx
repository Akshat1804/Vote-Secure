"use client"

// Update the ProfilePage to display registered voter data
import React, { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import { Download, Edit, Mail, MapPin, Phone, Shield, User } from "lucide-react"
import { useVoters } from "../../context/VoterContext"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

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

// Badge component
const Badge = React.forwardRef(({ className, variant, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variant === "default" && "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        variant === "secondary" && "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        variant === "success" && "border-transparent bg-green-100 text-green-800 hover:bg-green-200",
        variant === "destructive" &&
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        variant === "outline" && "text-foreground",
        variant === "warning" && "border-transparent bg-yellow-100 text-yellow-800 hover:bg-yellow-200",
        className,
      )}
      {...props}
    />
  )
})
Badge.displayName = "Badge"

export default function ProfilePage() {
  const [userData, setUserData] = useState(null)
  const { voters } = useVoters()
  const [currentVoter, setCurrentVoter] = useState(null)

  const [isElectionDay, setIsElectionDay] = useState(false)

  // Get user data from session storage and find the corresponding voter
  useEffect(() => {
    const storedUserData = sessionStorage.getItem("userData")
    if (storedUserData) {
      try {
        const parsedUserData = JSON.parse(storedUserData)
        setUserData(parsedUserData)

        // Find the voter with matching voter ID or Aadhaar
        const matchingVoter = voters.find(
          (voter) =>
            (voter.voterID && parsedUserData.voterID && voter.voterID === parsedUserData.voterID) ||
            (voter.aadhaar &&
              parsedUserData.aadhaar &&
              voter.aadhaar.replace(/\s/g, "") === parsedUserData.aadhaar.replace(/\s/g, "")),
        )

        if (matchingVoter) {
          // Create a complete voter object by merging matching voter with userData
          const completeVoter = {
            ...parsedUserData,
            ...matchingVoter,
            // Ensure we have the correct ID format
            id: matchingVoter.id || parsedUserData.voterID,
            voterID: matchingVoter.voterID || parsedUserData.voterID || matchingVoter.id,
            // Use the avatar from the matching voter if available, otherwise use from userData
            avatar: matchingVoter.avatar || parsedUserData.avatar,
          }
          setCurrentVoter(completeVoter)
        } else {
          // If no matching voter in voters array, use the userData as the voter data
          setCurrentVoter(parsedUserData)
        }
      } catch (error) {
        console.error("Error parsing user data:", error)
        // Handle corrupted data gracefully
        setUserData({
          name: "User",
          voterID: "Unknown",
          aadhaar: "",
          phone: "",
          email: "",
        })
      }
    }
  }, [voters])

  if (!userData) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600 mx-auto"></div>
          <p className="text-gray-600">Loading profile data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">My Profile</h1>
        <p className="text-gray-500">View and manage your personal information</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Profile Card */}
        <Card className="md:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div>
              <CardTitle>Voter Information</CardTitle>
              <CardDescription>Your personal details and voter information</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="flex flex-col md:flex-row">
              <div className="mb-6 flex flex-col items-center md:mb-0 md:w-1/3 md:border-r md:pr-6">
                {currentVoter?.avatar ? (
                  <div className="mb-4 h-32 w-32 overflow-hidden rounded-full bg-gradient-to-r from-blue-600 to-purple-600 p-1">
                    <img
                      src={currentVoter.avatar || "/placeholder.svg"}
                      alt={currentVoter.name}
                      className="h-full w-full rounded-full object-cover"
                      onError={(e) => {
                        e.target.onerror = null
                        e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(currentVoter.name)}&background=random&color=fff&size=128`
                      }}
                    />
                  </div>
                ) : (
                  <div className="mb-4 h-32 w-32 overflow-hidden rounded-full bg-gradient-to-r from-blue-600 to-purple-600 p-1">
                    <div className="flex h-full w-full items-center justify-center rounded-full bg-white">
                      <User className="h-16 w-16 text-gray-400" />
                    </div>
                  </div>
                )}
                <h3 className="text-xl font-bold text-gray-900">{currentVoter?.name || userData.name}</h3>
                <p className="text-sm text-gray-500">
                  Voter ID: {currentVoter?.id || currentVoter?.voterID || userData.voterID}
                </p>
                <div className="mt-4 flex space-x-2">
                  <Button variant="outline" size="sm" className="flex items-center gap-1">
                    <Download className="h-4 w-4" />
                    <span>Download ID</span>
                  </Button>
                </div>
              </div>

              <div className="md:w-2/3 md:pl-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Full Name</h4>
                    <div className="flex items-center justify-between">
                      <p className="text-gray-900">{currentVoter?.name || userData.name}</p>
                      <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                        <Edit className="h-3.5 w-3.5 text-gray-400" />
                      </Button>
                    </div>
                  </div>

                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Date of Birth</h4>
                    <p className="text-gray-900">{currentVoter?.dob || userData.dob}</p>
                  </div>
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Gender</h4>
                    <p className="text-gray-900">{currentVoter?.gender || userData.gender}</p>
                  </div>
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Aadhaar Number</h4>
                    <div className="flex items-center">
                      <p className="text-gray-900">{currentVoter?.aadhaar || userData.aadhaar}</p>
                      <Badge variant="success" className="ml-2">
                        Verified
                      </Badge>
                    </div>
                  </div>
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Phone Number</h4>
                    <div className="flex items-center justify-between">
                      <p className="text-gray-900">{currentVoter?.phone || userData.phone}</p>
                      <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                        <Edit className="h-3.5 w-3.5 text-gray-400" />
                      </Button>
                    </div>
                  </div>
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Email</h4>
                    <div className="flex items-center justify-between">
                      <p className="text-gray-900">{currentVoter?.email || userData.email}</p>
                      <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                        <Edit className="h-3.5 w-3.5 text-gray-400" />
                      </Button>
                    </div>
                  </div>
                  <div className="md:col-span-2">
                    <h4 className="mb-1 text-sm font-medium text-gray-500">Address</h4>
                    <div className="flex items-center justify-between">
                      <p className="text-gray-900">{currentVoter?.address || userData.address}</p>
                      <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                        <Edit className="h-3.5 w-3.5 text-gray-400" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Contact Information Card */}
        <Card>
          <CardHeader>
            <CardTitle>Contact Information</CardTitle>
            <CardDescription>Your contact details</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center rounded-md border border-gray-200 p-3">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                  <Phone className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700">Phone Number</h4>
                  <p className="text-gray-900">{currentVoter?.phone || userData.phone}</p>
                </div>
              </div>

              <div className="flex items-center rounded-md border border-gray-200 p-3">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                  <Mail className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700">Email Address</h4>
                  <p className="text-gray-900">{currentVoter?.email || userData.email}</p>
                </div>
              </div>

              <div className="flex items-center rounded-md border border-gray-200 p-3">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                  <MapPin className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700">Residential Address</h4>
                  <p className="text-gray-900">{currentVoter?.address || userData.address}</p>
                </div>
              </div>

              <div className="mt-4">
                <Link
                  to="/dashboard/settings"
                  className="inline-flex w-full items-center justify-center rounded-md bg-blue-600 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
                >
                  Update Contact Information
                </Link>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Verification Status Card */}
        <Card className="md:col-span-3">
          <CardHeader>
            <CardTitle>Verification Status</CardTitle>
            <CardDescription>Status of your identity verification</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex items-center justify-between rounded-md bg-green-50 p-3">
                <div className="flex items-center">
                  <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                    <Shield className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-green-800">Aadhaar Verification</h4>
                    <p className="text-sm text-green-700">Verified on {new Date().toLocaleDateString()}</p>
                  </div>
                </div>
                <Badge variant="success">Verified</Badge>
              </div>

              <div className="flex items-center justify-between rounded-md bg-green-50 p-3">
                <div className="flex items-center">
                  <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                    <User className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-green-800">Voter ID</h4>
                    <p className="text-sm text-green-700">Verified on {new Date().toLocaleDateString()}</p>
                  </div>
                </div>
                <Badge variant={currentVoter?.status === "Verified" ? "success" : "warning"}>
                  {currentVoter?.status || "Pending"}
                </Badge>
              </div>

              <div className="flex items-center justify-between rounded-md bg-yellow-50 p-3">
                <div className="flex items-center">
                  <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                    <MapPin className="h-5 w-5 text-yellow-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-yellow-800">Address Proof</h4>
                    <p className="text-sm text-yellow-700">Awaiting verification</p>
                  </div>
                </div>
                <Badge variant="warning">Pending</Badge>
              </div>
            </div>

            <div className="mt-6 rounded-lg border border-blue-100 bg-blue-50 p-4">
              <div className="flex items-start space-x-3">
                <Shield className="mt-0.5 h-5 w-5 text-blue-600" />
                <div>
                  <h4 className="text-sm font-medium text-blue-800">Verification Information</h4>
                  <p className="mt-1 text-xs text-blue-700">
                    Your identity has been verified using your Aadhaar details. To complete address verification, please
                    upload a valid address proof document in the Documents section.
                  </p>
                  <div className="mt-3">
                    <Link
                      to="/dashboard/documents"
                      className="inline-flex items-center text-xs font-medium text-blue-700 hover:text-blue-800 hover:underline"
                    >
                      Upload Address Proof
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

