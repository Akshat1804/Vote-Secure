"use client"

import React, { useState, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { AlertCircle, ArrowLeft, Check, Save, Upload, User, X, Search, Camera } from "lucide-react"
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

// Input component
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

// Label component
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

// Badge component
const Badge = React.forwardRef(({ className, variant, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variant === "success" && "border-green-200 bg-green-100 text-green-800",
        variant === "destructive" && "border-red-200 bg-red-100 text-red-800",
        variant === "warning" && "border-yellow-200 bg-yellow-100 text-yellow-800",
        variant === "info" && "border-blue-200 bg-blue-100 text-blue-800",
        className,
      )}
      {...props}
    />
  )
})
Badge.displayName = "Badge"

// Tabs components
const Tabs = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("w-full", className)} {...props} />
))
Tabs.displayName = "Tabs"

const TabsList = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
      className,
    )}
    {...props}
  />
))
TabsList.displayName = "TabsList"

const TabsTrigger = React.forwardRef(({ className, active, ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      active ? "bg-background text-foreground shadow-sm" : "hover:bg-background/50 hover:text-foreground",
      className,
    )}
    {...props}
  />
))
TabsTrigger.displayName = "TabsTrigger"

const TabsContent = React.forwardRef(({ className, active, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
      !active && "hidden",
      className,
    )}
    {...props}
  />
))
TabsContent.displayName = "TabsContent"

export default function RegisterUserPage() {
  const navigate = useNavigate()
  const { addVoter } = useVoters()

  const [formData, setFormData] = useState({
    aadhaarNumber: "",
    name: "",
    dob: "",
    gender: "",
    address: "",
    phone: "",
    email: "",
    voterID: "",
    constituency: "",
    pollingStation: "",
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)
  const [errors, setErrors] = useState({})
  const [activeTab, setActiveTab] = useState("details")
  const [photoPreview, setPhotoPreview] = useState(null)
  const [showPreview, setShowPreview] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const fileInputRef = useRef(null)
  const webcamRef = useRef(null)
  const [showCamera, setShowCamera] = useState(false)
  // Add a success state to track when a voter is successfully registered
  const [registrationSuccess, setRegistrationSuccess] = useState(false)
  const [registeredVoter, setRegisteredVoter] = useState(null)

  // Mock constituencies and polling stations
  const constituencies = [
    "Bangalore Central",
    "Bangalore North",
    "Bangalore South",
    "Mumbai North",
    "Mumbai South",
    "Mumbai Central",
    "Delhi East",
    "Delhi West",
    "Delhi North",
    "Delhi South",
    "Chennai Central",
    "Chennai North",
    "Chennai South",
  ]

  const pollingStations = {
    "Bangalore Central": ["St. Joseph's College", "Christ University", "Garuda Mall"],
    "Bangalore North": ["Hebbal School", "Yelahanka Community Hall", "Jakkur Flying School"],
    "Bangalore South": ["National College", "Jayanagar Complex", "BTM Layout Community Hall"],
    "Mumbai North": ["Borivali Station Hall", "Kandivali School", "Malad Community Center"],
    "Mumbai South": ["Churchgate Station Hall", "Marine Lines School", "Colaba Municipal School"],
    "Delhi East": ["Shahdara Community Hall", "Vivek Vihar School", "Anand Vihar Complex"],
    "Delhi West": ["Janakpuri Complex", "Dwarka Sector 10 Hall", "Rajouri Garden School"],
    "Chennai Central": ["Egmore Station Hall", "Nungambakkam School", "T Nagar Community Center"],
  }

  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: value,
    })

    // Clear error when user types
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: null,
      })
    }

    // Update polling stations when constituency changes
    if (name === "constituency") {
      setFormData((prev) => ({
        ...prev,
        pollingStation: "",
      }))
    }
  }

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
    setFormData({
      ...formData,
      aadhaarNumber: formatted,
    })

    // Clear error when user types
    if (errors.aadhaarNumber) {
      setErrors({
        ...errors,
        aadhaarNumber: null,
      })
    }
  }

  // Handle photo upload
  const handlePhotoUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setPhotoPreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  // Handle search
  const handleSearch = () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)

    // Simulate API call to search Aadhaar database
    setTimeout(() => {
      // Mock search results
      if (searchQuery.length >= 4) {
        const mockResults = [
          {
            aadhaarNumber: "1234 5678 9012",
            name: "Rahul Sharma",
            dob: "1985-06-15",
            gender: "Male",
            address: "123 MG Road, Bangalore, Karnataka 560001",
            phone: "9876543210",
            email: "rahul.sharma@example.com",
          },
          {
            aadhaarNumber: "2345 6789 0123",
            name: "Priya Patel",
            dob: "1990-03-22",
            gender: "Female",
            address: "456 Brigade Road, Bangalore, Karnataka 560001",
            phone: "8765432109",
            email: "priya.patel@example.com",
          },
        ]
        setSearchResults(mockResults)
      } else {
        setSearchResults([])
      }

      setIsSearching(false)
    }, 1500)
  }

  // Fill form with selected search result
  const fillFormWithResult = (result) => {
    setFormData({
      ...formData,
      aadhaarNumber: result.aadhaarNumber,
      name: result.name,
      dob: result.dob,
      gender: result.gender,
      address: result.address,
      phone: result.phone,
      email: result.email || "",
    })
    setSearchResults([])
    setSearchQuery("")
  }

  // Validate form
  const validateForm = () => {
    const newErrors = {}

    if (!formData.aadhaarNumber.replace(/\s/g, "") || formData.aadhaarNumber.replace(/\s/g, "").length !== 12) {
      newErrors.aadhaarNumber = "Valid 12-digit Aadhaar number is required"
    }

    if (!formData.name) {
      newErrors.name = "Name is required"
    }

    if (!formData.dob) {
      newErrors.dob = "Date of birth is required"
    }

    if (!formData.gender) {
      newErrors.gender = "Gender is required"
    }

    if (!formData.address) {
      newErrors.address = "Address is required"
    }

    if (!formData.phone) {
      newErrors.phone = "Phone number is required"
    }

    if (!formData.voterID) {
      newErrors.voterID = "Voter ID is required"
    }

    if (!formData.constituency) {
      newErrors.constituency = "Constituency is required"
    }

    if (!formData.pollingStation) {
      newErrors.pollingStation = "Polling station is required"
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault()

    if (validateForm()) {
      setIsSubmitting(true)

      // Create a new voter object
      const newVoter = {
        name: formData.name,
        aadhaar: formData.aadhaarNumber,
        dob: formData.dob,
        gender: formData.gender,
        address: formData.address,
        phone: formData.phone,
        email: formData.email || "",
        voterID: formData.voterID,
        constituency: formData.constituency,
        pollingStation: formData.pollingStation,
        // Save the photo preview as the avatar
        avatar:
          photoPreview ||
          `https://ui-avatars.com/api/?name=${encodeURIComponent(formData.name)}&background=random&color=fff&size=128`,
      }

      // Add the voter to the context
      const registeredVoter = addVoter(newVoter)
      setRegisteredVoter(registeredVoter)

      setTimeout(() => {
        setIsSubmitting(false)
        setShowSuccess(true)

        // Reset form after 3 seconds
        setTimeout(() => {
          setShowSuccess(false)
          setFormData({
            aadhaarNumber: "",
            name: "",
            dob: "",
            gender: "",
            address: "",
            phone: "",
            email: "",
            voterID: "",
            constituency: "",
            pollingStation: "",
          })
          setPhotoPreview(null)
          setActiveTab("details")

          // Navigate to voter management page
          navigate("/admin/voters")
        }, 3000)
      }, 1500)
    }
  }

  // Toggle preview mode
  const togglePreview = () => {
    if (!showPreview) {
      // Validate before showing preview
      if (validateForm()) {
        setShowPreview(true)
      }
    } else {
      setShowPreview(false)
    }
  }

  // Toggle camera
  const toggleCamera = () => {
    setShowCamera(!showCamera)
  }

  // Calculate age from DOB
  const calculateAge = (dob) => {
    if (!dob) return ""
    const birthDate = new Date(dob)
    const today = new Date()
    let age = today.getFullYear() - birthDate.getFullYear()
    const monthDiff = today.getMonth() - birthDate.getMonth()

    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--
    }

    return age
  }

  // Generate a random voter ID
  const generateVoterId = () => {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    const randomLetters =
      letters.charAt(Math.floor(Math.random() * letters.length)) +
      letters.charAt(Math.floor(Math.random() * letters.length)) +
      letters.charAt(Math.floor(Math.random() * letters.length))
    const randomNumbers = Math.floor(10000000 + Math.random() * 90000000)

    const voterId = randomLetters + randomNumbers

    setFormData({
      ...formData,
      voterID: voterId,
    })
  }

  return (
    <div className="p-4 md:p-6 bg-gray-50 min-h-screen">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Register New Voter</h1>
              <p className="text-gray-500">Admin panel for registering voters with Aadhaar details</p>
            </div>
            <Badge variant="destructive" className="px-3 py-1.5 text-sm">
              Admin Access Only
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left sidebar with instructions */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Registration Guide</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-start space-x-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                      <span className="text-xs font-bold">1</span>
                    </div>
                    <p className="text-sm text-gray-600">Search for citizen by Aadhaar number or name</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                      <span className="text-xs font-bold">2</span>
                    </div>
                    <p className="text-sm text-gray-600">Verify identity details from Aadhaar database</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                      <span className="text-xs font-bold">3</span>
                    </div>
                    <p className="text-sm text-gray-600">Assign voter ID and constituency details</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                      <span className="text-xs font-bold">4</span>
                    </div>
                    <p className="text-sm text-gray-600">Upload or capture photo ID for verification</p>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                      <span className="text-xs font-bold">5</span>
                    </div>
                    <p className="text-sm text-gray-600">Review and submit registration</p>
                  </div>
                </div>

                <div className="mt-6 rounded-lg bg-yellow-50 p-4">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 text-yellow-600" />
                    <div>
                      <h4 className="text-sm font-medium text-yellow-800">Important Notice</h4>
                      <p className="mt-1 text-sm text-yellow-700">
                        This form is for administrative use only. Ensure all details match the Aadhaar database exactly.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main content */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader className="border-b">
                <div className="flex justify-between items-center">
                  <CardTitle>Voter Registration Form</CardTitle>
                  <div className="flex space-x-2">
                    <Button type="button" variant="outline" size="sm" onClick={togglePreview}>
                      {showPreview ? "Edit Form" : "Preview"}
                    </Button>
                  </div>
                </div>
              </CardHeader>

              {!showPreview ? (
                <>
                  <CardContent className="p-6">
                    {/* Search section */}
                    <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                      <h3 className="text-sm font-medium text-blue-800 mb-2">Search Aadhaar Database</h3>
                      <div className="flex gap-2">
                        <div className="flex-1">
                          <Input
                            placeholder="Enter Aadhaar number or name"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                          />
                        </div>
                        <Button onClick={handleSearch} disabled={isSearching} className="bg-blue-600 hover:bg-blue-700">
                          {isSearching ? (
                            <>
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
                              Searching...
                            </>
                          ) : (
                            <>
                              <Search className="mr-2 h-4 w-4" />
                              Search
                            </>
                          )}
                        </Button>
                      </div>

                      {/* Search results */}
                      {searchResults.length > 0 && (
                        <div className="mt-3 border rounded-md bg-white">
                          <div className="p-2 text-xs font-medium text-gray-500 border-b">
                            {searchResults.length} results found
                          </div>
                          {searchResults.map((result, index) => (
                            <div
                              key={index}
                              className="p-3 border-b last:border-b-0 hover:bg-gray-50 cursor-pointer"
                              onClick={() => fillFormWithResult(result)}
                            >
                              <div className="flex justify-between">
                                <div>
                                  <p className="font-medium">{result.name}</p>
                                  <p className="text-sm text-gray-500">Aadhaar: {result.aadhaarNumber}</p>
                                </div>
                                <Button variant="ghost" size="sm">
                                  Select
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Tabs */}
                    <Tabs>
                      <TabsList className="w-full grid grid-cols-2 mb-6">
                        <TabsTrigger active={activeTab === "details"} onClick={() => setActiveTab("details")}>
                          Personal Details
                        </TabsTrigger>
                        <TabsTrigger active={activeTab === "voting"} onClick={() => setActiveTab("voting")}>
                          Voting Information
                        </TabsTrigger>
                      </TabsList>

                      {/* Personal Details Tab */}
                      <TabsContent active={activeTab === "details"}>
                        <form>
                          <div className="grid gap-6 md:grid-cols-2">
                            <div className="md:col-span-2">
                              <Label htmlFor="aadhaarNumber">Aadhaar Number *</Label>
                              <Input
                                id="aadhaarNumber"
                                name="aadhaarNumber"
                                placeholder="XXXX XXXX XXXX"
                                value={formData.aadhaarNumber}
                                onChange={handleAadhaarChange}
                                maxLength={14}
                                className={errors.aadhaarNumber ? "border-red-300" : ""}
                              />
                              {errors.aadhaarNumber && (
                                <p className="mt-1 text-xs text-red-500">{errors.aadhaarNumber}</p>
                              )}
                            </div>

                            <div>
                              <Label htmlFor="name">Full Name *</Label>
                              <Input
                                id="name"
                                name="name"
                                value={formData.name}
                                onChange={handleChange}
                                className={errors.name ? "border-red-300" : ""}
                              />
                              {errors.name && <p className="mt-1 text-xs text-red-500">{errors.name}</p>}
                            </div>

                            <div>
                              <Label htmlFor="dob">Date of Birth *</Label>
                              <Input
                                id="dob"
                                name="dob"
                                type="date"
                                value={formData.dob}
                                onChange={handleChange}
                                className={errors.dob ? "border-red-300" : ""}
                              />
                              {errors.dob && <p className="mt-1 text-xs text-red-500">{errors.dob}</p>}
                              {formData.dob && (
                                <p className="mt-1 text-xs text-gray-500">Age: {calculateAge(formData.dob)} years</p>
                              )}
                            </div>

                            <div>
                              <Label htmlFor="gender">Gender *</Label>
                              <select
                                id="gender"
                                name="gender"
                                value={formData.gender}
                                onChange={handleChange}
                                className={cn(
                                  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                                  errors.gender ? "border-red-300" : "",
                                )}
                              >
                                <option value="">Select gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                                <option value="Other">Other</option>
                              </select>
                              {errors.gender && <p className="mt-1 text-xs text-red-500">{errors.gender}</p>}
                            </div>

                            <div className="md:col-span-2">
                              <Label htmlFor="address">Address *</Label>
                              <textarea
                                id="address"
                                name="address"
                                rows={3}
                                value={formData.address}
                                onChange={handleChange}
                                className={cn(
                                  "flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                                  errors.address ? "border-red-300" : "",
                                )}
                              />
                              {errors.address && <p className="mt-1 text-xs text-red-500">{errors.address}</p>}
                            </div>

                            <div>
                              <Label htmlFor="phone">Phone Number *</Label>
                              <Input
                                id="phone"
                                name="phone"
                                value={formData.phone}
                                onChange={handleChange}
                                className={errors.phone ? "border-red-300" : ""}
                              />
                              {errors.phone && <p className="mt-1 text-xs text-red-500">{errors.phone}</p>}
                            </div>

                            <div>
                              <Label htmlFor="email">Email Address</Label>
                              <Input
                                id="email"
                                name="email"
                                type="email"
                                value={formData.email}
                                onChange={handleChange}
                              />
                            </div>
                          </div>

                          <div className="mt-6 flex justify-end">
                            <Button
                              type="button"
                              onClick={() => setActiveTab("voting")}
                              className="bg-blue-600 hover:bg-blue-700"
                            >
                              Next: Voting Information
                            </Button>
                          </div>
                        </form>
                      </TabsContent>

                      {/* Voting Information Tab */}
                      <TabsContent active={activeTab === "voting"}>
                        <form>
                          <div className="grid gap-6 md:grid-cols-2">
                            <div>
                              <Label htmlFor="voterID">Voter ID *</Label>
                              <div className="flex gap-2">
                                <Input
                                  id="voterID"
                                  name="voterID"
                                  value={formData.voterID}
                                  onChange={handleChange}
                                  className={errors.voterID ? "border-red-300" : ""}
                                />
                                <Button
                                  type="button"
                                  variant="outline"
                                  onClick={generateVoterId}
                                  className="whitespace-nowrap"
                                >
                                  Generate ID
                                </Button>
                              </div>
                              {errors.voterID && <p className="mt-1 text-xs text-red-500">{errors.voterID}</p>}
                            </div>

                            <div>
                              <Label htmlFor="constituency">Constituency *</Label>
                              <select
                                id="constituency"
                                name="constituency"
                                value={formData.constituency}
                                onChange={handleChange}
                                className={cn(
                                  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                                  errors.constituency ? "border-red-300" : "",
                                )}
                              >
                                <option value="">Select constituency</option>
                                {constituencies.map((constituency) => (
                                  <option key={constituency} value={constituency}>
                                    {constituency}
                                  </option>
                                ))}
                              </select>
                              {errors.constituency && (
                                <p className="mt-1 text-xs text-red-500">{errors.constituency}</p>
                              )}
                            </div>

                            <div>
                              <Label htmlFor="pollingStation">Polling Station *</Label>
                              <select
                                id="pollingStation"
                                name="pollingStation"
                                value={formData.pollingStation}
                                onChange={handleChange}
                                disabled={!formData.constituency}
                                className={cn(
                                  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
                                  errors.pollingStation ? "border-red-300" : "",
                                )}
                              >
                                <option value="">Select polling station</option>
                                {formData.constituency &&
                                  pollingStations[formData.constituency]?.map((station) => (
                                    <option key={station} value={station}>
                                      {station}
                                    </option>
                                  ))}
                              </select>
                              {errors.pollingStation && (
                                <p className="mt-1 text-xs text-red-500">{errors.pollingStation}</p>
                              )}
                            </div>

                            <div className="md:col-span-2">
                              <Label htmlFor="photo">Photo ID</Label>
                              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 bg-gray-50">
                                  {photoPreview ? (
                                    <div className="relative w-full">
                                      <img
                                        src={photoPreview || "/placeholder.svg"}
                                        alt="Preview"
                                        className="mx-auto h-40 object-cover rounded-md"
                                      />
                                      <button
                                        type="button"
                                        onClick={() => setPhotoPreview(null)}
                                        className="absolute top-0 right-0 bg-red-500 text-white rounded-full p-1"
                                      >
                                        <X className="h-4 w-4" />
                                      </button>
                                    </div>
                                  ) : (
                                    <>
                                      <User className="h-10 w-10 text-gray-400 mb-2" />
                                      <p className="text-sm text-gray-500 mb-4">Upload voter photo</p>
                                      <input
                                        ref={fileInputRef}
                                        id="photo-upload"
                                        type="file"
                                        className="hidden"
                                        accept="image/*"
                                        onChange={handlePhotoUpload}
                                      />
                                      <Button
                                        type="button"
                                        variant="outline"
                                        onClick={() => fileInputRef.current?.click()}
                                      >
                                        <Upload className="mr-2 h-4 w-4" />
                                        Upload Photo
                                      </Button>
                                    </>
                                  )}
                                </div>

                                <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 bg-gray-50">
                                  {showCamera ? (
                                    <div className="relative w-full">
                                      <div className="bg-black rounded-md h-40 flex items-center justify-center">
                                        <p className="text-white">Camera would appear here</p>
                                      </div>
                                      <div className="mt-2 flex justify-center">
                                        <Button type="button" variant="outline" onClick={toggleCamera}>
                                          Close Camera
                                        </Button>
                                      </div>
                                    </div>
                                  ) : (
                                    <>
                                      <Camera className="h-10 w-10 text-gray-400 mb-2" />
                                      <p className="text-sm text-gray-500 mb-4">Take a photo now</p>
                                      <Button type="button" variant="outline" onClick={toggleCamera}>
                                        <Camera className="mr-2 h-4 w-4" />
                                        Open Camera
                                      </Button>
                                    </>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>

                          <div className="mt-6 flex justify-between">
                            <Button type="button" variant="outline" onClick={() => setActiveTab("details")}>
                              <ArrowLeft className="mr-2 h-4 w-4" />
                              Back to Personal Details
                            </Button>
                            <Button type="button" onClick={togglePreview} className="bg-blue-600 hover:bg-blue-700">
                              Preview Registration
                            </Button>
                          </div>
                        </form>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </>
              ) : (
                <CardContent className="p-6">
                  <div className="bg-blue-50 p-4 rounded-lg mb-6">
                    <h3 className="text-sm font-medium text-blue-800 mb-2">Registration Preview</h3>
                    <p className="text-sm text-blue-600">Please review all information before submitting</p>
                  </div>

                  <div className="flex justify-center mb-6">
                    <div className="w-full max-w-md bg-white rounded-lg border border-gray-200 shadow-md overflow-hidden">
                      <div className="p-6 flex flex-col items-center">
                        {/* Photo Preview */}
                        <div className="mb-4">
                          {photoPreview ? (
                            <img
                              src={photoPreview || "/placeholder.svg"}
                              alt="Voter Photo"
                              className="h-32 w-32 rounded-full object-cover border-2 border-gray-200"
                            />
                          ) : (
                            <div className="h-32 w-32 rounded-full bg-gray-200 flex items-center justify-center text-gray-600 text-4xl font-medium border-2 border-gray-200">
                              {formData.name ? formData.name.charAt(0) : "U"}
                            </div>
                          )}
                        </div>

                        {/* Voter ID and Name */}
                        <h3 className="text-xl font-bold text-gray-900">{formData.name}</h3>
                        <div className="mt-1 px-3 py-1 bg-blue-100 rounded-full text-blue-800 text-sm font-medium">
                          {formData.voterID}
                        </div>

                        {/* Basic Info */}
                        <div className="w-full mt-6 grid grid-cols-2 gap-4">
                          <div className="col-span-2">
                            <h4 className="text-sm font-medium text-gray-500 mb-1">Address</h4>
                            <p className="text-sm text-gray-900 bg-gray-50 p-2 rounded border border-gray-100">
                              {formData.address}
                            </p>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-500 mb-1">Date of Birth</h4>
                            <p className="text-sm text-gray-900">
                              {formData.dob} ({calculateAge(formData.dob)} years)
                            </p>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-500 mb-1">Gender</h4>
                            <p className="text-sm text-gray-900">{formData.gender}</p>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-500 mb-1">Phone</h4>
                            <p className="text-sm text-gray-900">{formData.phone}</p>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-500 mb-1">Email</h4>
                            <p className="text-sm text-gray-900">{formData.email || "Not provided"}</p>
                          </div>
                        </div>
                      </div>

                      {/* Voting Information */}
                      <div className="bg-gray-50 border-t border-gray-200 p-6">
                        <h4 className="font-medium text-gray-900 mb-3">Voting Information</h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <h5 className="text-sm font-medium text-gray-500 mb-1">Aadhaar Number</h5>
                            <p className="text-sm text-gray-900">{formData.aadhaarNumber}</p>
                          </div>
                          <div>
                            <h5 className="text-sm font-medium text-gray-500 mb-1">Constituency</h5>
                            <p className="text-sm text-gray-900">{formData.constituency}</p>
                          </div>
                          <div className="col-span-2">
                            <h5 className="text-sm font-medium text-gray-500 mb-1">Polling Station</h5>
                            <p className="text-sm text-gray-900">{formData.pollingStation}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 flex justify-between">
                    <Button type="button" variant="outline" onClick={togglePreview}>
                      <ArrowLeft className="mr-2 h-4 w-4" />
                      Back to Edit
                    </Button>
                    <Button
                      type="button"
                      onClick={handleSubmit}
                      disabled={isSubmitting}
                      className="bg-green-600 hover:bg-green-700"
                    >
                      {isSubmitting ? (
                        <>
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
                          Registering...
                        </>
                      ) : (
                        <>
                          <Save className="mr-2 h-4 w-4" />
                          Submit Registration
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              )}
            </Card>
          </div>
        </div>
      </div>

      {/* Success Message */}
      {showSuccess && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4 animate-in fade-in slide-in-from-bottom-10 duration-300">
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-green-100 mx-auto mb-4">
              <Check className="h-6 w-6 text-green-600" />
            </div>
            <h3 className="text-lg font-medium text-center text-gray-900 mb-2">Registration Successful!</h3>
            <p className="text-center text-gray-500 mb-6">New voter has been registered and is pending verification.</p>
            <div className="bg-gray-50 p-4 rounded-md mb-4">
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium text-gray-500">Voter ID:</span>
                <span className="text-sm font-medium text-gray-900">{registeredVoter?.id || formData.voterID}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium text-gray-500">Name:</span>
                <span className="text-sm font-medium text-gray-900">{formData.name}</span>
              </div>
            </div>
            <div className="flex justify-center">
              <Button onClick={() => setShowSuccess(false)} className="w-full bg-green-600 hover:bg-green-700">
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

