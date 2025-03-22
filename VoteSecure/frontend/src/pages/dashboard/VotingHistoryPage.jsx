"use client"

import React, { useState } from "react"
import { AlertCircle, Calendar, Clock, Download, FileText, MapPin, PieChart, QrCode, Shield } from "lucide-react"

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

export default function VotingHistoryPage() {
  const [filterYear, setFilterYear] = useState("all")
  const [filterType, setFilterType] = useState("all")

  // Mock voting history data
  const votingHistory = [
    {
      id: 1,
      election: "General Elections 2024",
      date: "April 12, 2024",
      type: "National",
      status: "Voted",
      booth: "Government High School, Sector 12",
      time: "10:45 AM",
      verification: "QR Code + Biometric",
      year: "2024",
    },
    {
      id: 2,
      election: "Municipal Corporation Elections",
      date: "June 5, 2023",
      type: "Local",
      status: "Voted",
      booth: "Primary School, Sector 8",
      time: "2:30 PM",
      verification: "QR Code + Biometric",
      year: "2023",
    },
    {
      id: 3,
      election: "State Assembly Elections",
      date: "November 20, 2022",
      type: "State",
      status: "Not Voted",
      booth: "-",
      time: "-",
      verification: "-",
      year: "2022",
    },
  ]

  // Filter voting history based on year and type
  const filteredHistory = votingHistory.filter((history) => {
    const matchesYear = filterYear === "all" || history.year === filterYear
    const matchesType = filterType === "all" || history.type.toLowerCase() === filterType.toLowerCase()
    return matchesYear && matchesType
  })

  // Calculate participation statistics
  const totalElections = votingHistory.length
  const votedElections = votingHistory.filter((history) => history.status === "Voted").length
  const participationRate = totalElections > 0 ? Math.round((votedElections / totalElections) * 100) : 0

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Voting History</h1>
        <p className="text-gray-500">View your past voting participation and records</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Voting Summary Card */}
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
          <CardHeader>
            <CardTitle>Voting Summary</CardTitle>
            <CardDescription>Your participation in elections</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="rounded-lg bg-white p-3 shadow-sm">
                <p className="text-2xl font-bold text-blue-600">{totalElections}</p>
                <p className="text-xs text-gray-500">Elections</p>
              </div>
              <div className="rounded-lg bg-white p-3 shadow-sm">
                <p className="text-2xl font-bold text-green-600">{votedElections}</p>
                <p className="text-xs text-gray-500">Voted</p>
              </div>
              <div className="rounded-lg bg-white p-3 shadow-sm">
                <p className="text-2xl font-bold text-gray-600">{participationRate}%</p>
                <p className="text-xs text-gray-500">Participation</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Participation Trend Card */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Participation Trend</CardTitle>
            <CardDescription>Your voting history over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex h-48 w-full items-center justify-center">
              <div className="relative h-32 w-32">
                <svg className="h-32 w-32 -rotate-90 transform">
                  <circle cx="64" cy="64" r="56" stroke="#e2e8f0" strokeWidth="16" fill="none" />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="url(#participation-gradient)"
                    strokeWidth="16"
                    fill="none"
                    strokeDasharray={2 * Math.PI * 56}
                    strokeDashoffset={2 * Math.PI * 56 * (1 - participationRate / 100)}
                    strokeLinecap="round"
                  />
                  <defs>
                    <linearGradient id="participation-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#3B82F6" />
                      <stop offset="100%" stopColor="#8B5CF6" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-3xl font-bold text-blue-600">{participationRate}%</span>
                </div>
              </div>
              <div className="ml-8 space-y-4">
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Your participation</span>
                    <span className="font-medium text-blue-600">{participationRate}%</span>
                  </div>
                  <Progress value={participationRate} max={100} className="h-2" />
                </div>
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">National average</span>
                    <span className="font-medium text-gray-600">62%</span>
                  </div>
                  <Progress value={62} max={100} className="h-2 bg-gray-200" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="mb-6 mt-8 flex flex-wrap items-center gap-4">
        <div>
          <label htmlFor="year-filter" className="mr-2 text-sm font-medium text-gray-700">
            Filter by Year:
          </label>
          <select
            id="year-filter"
            value={filterYear}
            onChange={(e) => setFilterYear(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All Years</option>
            <option value="2024">2024</option>
            <option value="2023">2023</option>
            <option value="2022">2022</option>
          </select>
        </div>
        <div>
          <label htmlFor="type-filter" className="mr-2 text-sm font-medium text-gray-700">
            Filter by Type:
          </label>
          <select
            id="type-filter"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All Types</option>
            <option value="national">National</option>
            <option value="state">State</option>
            <option value="local">Local</option>
          </select>
        </div>
        <Button variant="outline" size="sm" className="ml-auto">
          <Download className="mr-1.5 h-4 w-4" />
          Export History
        </Button>
      </div>

      {/* Voting History List */}
      <Card>
        <CardHeader>
          <CardTitle>Voting History</CardTitle>
          <CardDescription>Record of your past voting participation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {filteredHistory.length > 0 ? (
              filteredHistory.map((history) => (
                <div key={history.id} className="rounded-lg border">
                  <div className="border-b bg-gradient-to-r from-blue-50 to-purple-50 px-4 py-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-gray-900">{history.election}</h4>
                      <Badge variant={history.status === "Voted" ? "success" : "destructive"}>{history.status}</Badge>
                    </div>
                  </div>
                  <div className="p-4">
                    {history.status === "Voted" ? (
                      <>
                        <div className="flex flex-col space-y-4 sm:flex-row sm:space-x-6 sm:space-y-0">
                          <div className="sm:w-1/3">
                            <h5 className="text-sm font-medium text-gray-500">Election Details</h5>
                            <div className="mt-2 space-y-2">
                              <div className="flex items-center text-sm">
                                <Calendar className="mr-2 h-4 w-4 text-gray-400" />
                                <span>Date: {history.date}</span>
                              </div>
                              <div className="flex items-center text-sm">
                                <MapPin className="mr-2 h-4 w-4 text-gray-400" />
                                <span>Booth: {history.booth}</span>
                              </div>
                            </div>
                          </div>

                          <div className="sm:w-1/3">
                            <h5 className="text-sm font-medium text-gray-500">Voting Time</h5>
                            <div className="mt-2 space-y-2">
                              <div className="flex items-center text-sm">
                                <Clock className="mr-2 h-4 w-4 text-gray-400" />
                                <span>{history.time}</span>
                              </div>
                            </div>
                          </div>

                          <div className="sm:w-1/3">
                            <h5 className="text-sm font-medium text-gray-500">Verification Method</h5>
                            <div className="mt-2 space-y-2">
                              <div className="flex items-center text-sm">
                                <QrCode className="mr-2 h-4 w-4 text-gray-400" />
                                <span>{history.verification}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="mt-4 rounded-lg bg-green-50 p-3">
                          <div className="flex items-center">
                            <Shield className="mr-2 h-5 w-5 text-green-600" />
                            <span className="text-sm font-medium text-green-800">
                              Your vote was securely recorded. Thank you for participating!
                            </span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="flex flex-col space-y-4 sm:flex-row sm:space-x-6 sm:space-y-0">
                        <div className="sm:w-1/3">
                          <h5 className="text-sm font-medium text-gray-500">Election Details</h5>
                          <div className="mt-2 space-y-2">
                            <div className="flex items-center text-sm">
                              <Calendar className="mr-2 h-4 w-4 text-gray-400" />
                              <span>Date: {history.date}</span>
                            </div>
                          </div>
                        </div>

                        <div className="sm:w-2/3">
                          <div className="mt-4 rounded-lg bg-red-50 p-3">
                            <div className="flex items-center">
                              <AlertCircle className="mr-2 h-5 w-5 text-red-600" />
                              <span className="text-sm font-medium text-red-800">
                                You did not participate in this election.
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
                  <FileText className="h-6 w-6 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">No voting history found</h3>
                <p className="mt-2 text-sm text-gray-500">
                  No voting records match your filter criteria. Try adjusting your filters or check back after
                  participating in an election.
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Historical Election Data */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Historical Election Data</CardTitle>
          <CardDescription>Voting trends over the past elections</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              {
                year: "2024",
                turnout: "68.5%",
                registered: "245,678",
                trend: "up",
              },
              {
                year: "2020",
                turnout: "63.2%",
                registered: "238,142",
                trend: "up",
              },
              {
                year: "2016",
                turnout: "61.8%",
                registered: "225,890",
                trend: "down",
              },
            ].map((election, index) => (
              <div key={index} className="rounded-lg border p-3">
                <h4 className="font-medium text-gray-900">{election.year} Elections</h4>
                <div className="mt-2 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-500">Voter Turnout:</span>
                    <span className="text-sm font-medium">{election.turnout}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-500">Registered Voters:</span>
                    <span className="text-sm font-medium">{election.registered}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-500">Trend:</span>
                    <Badge variant={election.trend === "up" ? "success" : "destructive"}>
                      {election.trend === "up" ? "Increasing" : "Decreasing"}
                    </Badge>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-lg bg-blue-50 p-4">
            <div className="flex items-start space-x-3">
              <PieChart className="mt-0.5 h-5 w-5 text-blue-600" />
              <div>
                <h4 className="text-sm font-medium text-blue-800">Voting Impact</h4>
                <p className="mt-1 text-sm text-blue-700">
                  Your vote matters! In the 2020 local elections, several seats were decided by less than 100 votes.
                  Participating in elections ensures your voice is heard in shaping policies that affect your community.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

