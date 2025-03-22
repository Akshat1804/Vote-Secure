"use client"

import React, { useState, useEffect } from "react"
import { useParams, Link } from "react-router-dom"
import { AlertCircle, ArrowLeft, Calendar, Clock, Download, FileText, Info, MapPin, Users, Check } from "lucide-react"

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

// Progress component
const Progress = React.forwardRef(({ className, value, max = 100, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
      {...props}
    >
      <div
        className="h-full w-full flex-1 bg-primary transition-all"
        style={{ transform: `translateX(-${100 - (value / max) * 100}%)` }}
      />
    </div>
  )
})
Progress.displayName = "Progress"

export default function ElectionDetailsPage() {
  const { id } = useParams()
  const [election, setElection] = useState(null)
  const [loading, setLoading] = useState(true)

  // Mock election data
  const electionData = {
    ge2025: {
      id: "ge2025",
      title: "General Elections 2025",
      date: "April 15, 2025",
      daysLeft: 28,
      type: "National",
      status: "Upcoming",
      description: "Parliamentary elections to elect members of the Lok Sabha",
      location: "Your assigned polling booth",
      importance: "high",
      votingHours: "8:00 AM - 6:00 PM",
      eligibility: "All citizens aged 18 and above with valid voter ID",
      candidates: [
        { name: "Candidate A", party: "Party X", symbol: "🌟" },
        { name: "Candidate B", party: "Party Y", symbol: "🌱" },
        { name: "Candidate C", party: "Party Z", symbol: "🔔" },
      ],
      documents: ["Voter ID Card", "Aadhaar Card", "QR Code"],
      pollingStation: {
        name: "Government High School",
        address: "Plot No. 15, Sector 12, Gandhinagar, Gujarat - 382016",
        boothNumber: "42",
      },
      phases: [
        { phase: "Phase 1", date: "April 15, 2025", states: "Gujarat, Maharashtra, Karnataka" },
        { phase: "Phase 2", date: "April 22, 2025", states: "Tamil Nadu, Kerala, Andhra Pradesh" },
        { phase: "Phase 3", date: "April 29, 2025", states: "Uttar Pradesh, Bihar, West Bengal" },
      ],
    },
    mc2025: {
      id: "mc2025",
      title: "Municipal Corporation Elections",
      date: "June 10, 2025",
      daysLeft: 84,
      type: "Local",
      status: "Upcoming",
      description: "Elections to elect members of the Municipal Corporation",
      location: "Your assigned polling booth",
      importance: "medium",
      votingHours: "8:00 AM - 6:00 PM",
      eligibility: "All citizens aged 18 and above with valid voter ID residing in the municipal area",
      candidates: [
        { name: "Candidate D", party: "Party X", symbol: "🌟" },
        { name: "Candidate E", party: "Party Y", symbol: "🌱" },
        { name: "Candidate F", party: "Independent", symbol: "🔔" },
      ],
      documents: ["Voter ID Card", "Aadhaar Card", "QR Code"],
      pollingStation: {
        name: "Primary School",
        address: "Plot No. 8, Sector 8, Gandhinagar, Gujarat - 382008",
        boothNumber: "15",
      },
      phases: [{ phase: "Single Phase", date: "June 10, 2025", states: "All Municipal Areas" }],
    },
  }

  // Fetch election data
  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      if (electionData[id]) {
        setElection(electionData[id])
      }
      setLoading(false)
    }, 500)
  }, [id])

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600 mx-auto"></div>
          <p className="text-gray-600">Loading election details...</p>
        </div>
      </div>
    )
  }

  if (!election) {
    return (
      <div className="p-4 md:p-6">
        <div className="mb-6">
          <Link to="/dashboard/elections" className="inline-flex items-center text-blue-600 hover:text-blue-800">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Elections
          </Link>
        </div>
        <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center">
          <AlertCircle className="mx-auto mb-4 h-12 w-12 text-red-500" />
          <h2 className="text-xl font-bold text-gray-900">Election Not Found</h2>
          <p className="mt-2 text-gray-600">The election you're looking for doesn't exist or has been removed.</p>
          <Button className="mt-4" asChild>
            <Link to="/dashboard/elections">View All Elections</Link>
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <Link to="/dashboard/elections" className="inline-flex items-center text-blue-600 hover:text-blue-800">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Elections
        </Link>
      </div>

      {/* Election Header */}
      <div className="mb-6 overflow-hidden rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white shadow-lg">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <div className="flex items-center">
              <Badge variant="warning" className="mb-2">
                {election.type}
              </Badge>
            </div>
            <h1 className="text-2xl font-bold">{election.title}</h1>
            <p className="mt-1 text-blue-100">{election.description}</p>
            <div className="mt-4 flex flex-wrap items-center gap-4">
              <div className="flex items-center">
                <Calendar className="mr-2 h-5 w-5" />
                <span>{election.date}</span>
              </div>
              <div className="flex items-center">
                <Clock className="mr-2 h-5 w-5" />
                <span>{election.votingHours}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/20 text-2xl font-bold backdrop-blur-sm">
                {election.daysLeft}
              </div>
              <span className="mt-1 text-sm">Days Left</span>
            </div>
            <Link
              to="/dashboard/polling-map"
              className="rounded-md bg-white px-4 py-2 text-sm font-medium text-blue-700 shadow-sm transition-colors hover:bg-white/90"
            >
              Find Polling Station
            </Link>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Election Details */}
        <div className="md:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Election Details</CardTitle>
              <CardDescription>Information about the upcoming election</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">About this Election</h3>
                  <p className="text-gray-700">
                    {election.title} is scheduled for {election.date}. This is a {election.type.toLowerCase()} level
                    election where eligible voters will cast their votes to elect representatives. The election will be
                    conducted using Electronic Voting Machines (EVMs) with Voter Verifiable Paper Audit Trail (VVPAT)
                    systems.
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">Election Schedule</h3>
                  <div className="rounded-lg border">
                    <div className="border-b bg-gray-50 px-4 py-2">
                      <div className="grid grid-cols-3 font-medium text-gray-700">
                        <div>Phase</div>
                        <div>Date</div>
                        <div>States/Regions</div>
                      </div>
                    </div>
                    <div className="divide-y">
                      {election.phases.map((phase, index) => (
                        <div key={index} className="grid grid-cols-3 px-4 py-3 text-sm">
                          <div>{phase.phase}</div>
                          <div>{phase.date}</div>
                          <div>{phase.states}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="rounded-lg bg-blue-50 p-4">
                  <div className="flex items-start space-x-3">
                    <Info className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-800">Voter Eligibility</h4>
                      <p className="mt-1 text-sm text-blue-700">{election.eligibility}</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">Required Documents</h3>
                  <div className="flex flex-wrap gap-2">
                    {election.documents.map((document, index) => (
                      <div
                        key={index}
                        className="flex items-center rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm"
                      >
                        <FileText className="mr-2 h-4 w-4 text-blue-600" />
                        {document}
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="mb-2 text-lg font-medium text-gray-900">Candidates</h3>
                  <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
                    {election.candidates.map((candidate, index) => (
                      <div key={index} className="rounded-lg border p-4">
                        <div className="flex items-center space-x-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-100 text-xl">
                            {candidate.symbol}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{candidate.name}</h4>
                            <p className="text-sm text-gray-500">{candidate.party}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-lg bg-yellow-50 p-4">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 text-yellow-600" />
                    <div>
                      <h4 className="text-sm font-medium text-yellow-800">Important Notice</h4>
                      <p className="mt-1 text-sm text-yellow-700">
                        Carrying mobile phones, cameras, or any electronic devices inside the polling booth is strictly
                        prohibited. Please ensure you bring all required identification documents on election day.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Your Polling Station */}
          <Card>
            <CardHeader>
              <CardTitle>Your Polling Station</CardTitle>
              <CardDescription>Where you'll cast your vote</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 p-4">
                <div className="flex items-center space-x-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                    <MapPin className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">{election.pollingStation.name}</h4>
                    <p className="text-sm text-gray-600">Booth #{election.pollingStation.boothNumber}</p>
                  </div>
                </div>
                <div className="mt-3 rounded-md bg-white p-3 shadow-sm">
                  <p className="text-sm text-gray-700">{election.pollingStation.address}</p>
                </div>
              </div>

              <div className="mt-4">
                <Link
                  to="/dashboard/polling-map"
                  className="inline-flex w-full items-center justify-center rounded-md border border-gray-200 bg-white py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
                >
                  View on Map
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Voting Checklist */}
          <Card>
            <CardHeader>
              <CardTitle>Voting Checklist</CardTitle>
              <CardDescription>Prepare for election day</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { text: "Check your voter registration status", completed: true },
                  { text: "Verify your polling station location", completed: true },
                  { text: "Prepare required identification documents", completed: false },
                  { text: "Download your QR code", completed: false },
                  { text: "Plan your visit during voting hours", completed: false },
                ].map((item, index) => (
                  <div
                    key={index}
                    className={`flex items-center rounded-md p-2 ${item.completed ? "bg-green-50" : "bg-gray-50"}`}
                  >
                    <div
                      className={`mr-3 flex h-6 w-6 items-center justify-center rounded-full ${
                        item.completed ? "bg-green-500 text-white" : "bg-white"
                      }`}
                    >
                      {item.completed ? <Check className="h-3.5 w-3.5" /> : index + 1}
                    </div>
                    <span className={`text-sm ${item.completed ? "text-green-800" : "text-gray-700"}`}>
                      {item.text}
                    </span>
                  </div>
                ))}
              </div>

              <div className="mt-4">
                <Button className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Download Voter Guide
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Voter Turnout */}
          <Card>
            <CardHeader>
              <CardTitle>Previous Turnout</CardTitle>
              <CardDescription>Historical voter participation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="text-gray-600">2020 Elections</span>
                    <span className="font-medium text-blue-600">67%</span>
                  </div>
                  <Progress value={67} max={100} className="h-2" />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="text-gray-600">2016 Elections</span>
                    <span className="font-medium text-blue-600">62%</span>
                  </div>
                  <Progress value={62} max={100} className="h-2" />
                </div>
                <div>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="text-gray-600">2012 Elections</span>
                    <span className="font-medium text-blue-600">58%</span>
                  </div>
                  <Progress value={58} max={100} className="h-2" />
                </div>

                <div className="flex items-center justify-between rounded-md bg-blue-50 p-3">
                  <div className="flex items-center">
                    <Users className="mr-2 h-5 w-5 text-blue-600" />
                    <span className="text-sm font-medium text-blue-800">Target Turnout</span>
                  </div>
                  <span className="text-sm font-bold text-blue-800">75%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Call to Action */}
      <div className="mt-6 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white shadow-lg">
        <div className="flex flex-col items-center justify-between gap-4 text-center md:flex-row md:text-left">
          <div>
            <h2 className="text-xl font-bold">Ready to make your voice heard?</h2>
            <p className="mt-1 text-blue-100">
              Your vote matters! Prepare for the upcoming election and be part of the democratic process.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/dashboard/polling-map"
              className="rounded-md bg-white px-4 py-2 text-sm font-medium text-blue-700 shadow-sm transition-colors hover:bg-white/90"
            >
              <MapPin className="mr-2 h-4 w-4" />
              Find Your Polling Station
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}

