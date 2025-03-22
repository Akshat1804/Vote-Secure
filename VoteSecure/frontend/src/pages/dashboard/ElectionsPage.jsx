"use client"

import React, { useState } from "react"
import { Link } from "react-router-dom"
import { AlertCircle, Calendar, ChevronRight, Clock, Filter, MapPin, Search, Shield } from "lucide-react"

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

export default function ElectionsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [filterType, setFilterType] = useState("all")

  // Mock election data
  const elections = [
    {
      id: "ge2025",
      title: "General Elections 2025",
      date: "April 15, 2025",
      daysLeft: 28,
      type: "National",
      status: "Upcoming",
      description: "Parliamentary elections to elect members of the Lok Sabha",
      location: "Your assigned polling booth",
      importance: "high",
    },
    {
      id: "mc2025",
      title: "Municipal Corporation Elections",
      date: "June 10, 2025",
      daysLeft: 84,
      type: "Local",
      status: "Upcoming",
      description: "Elections to elect members of the Municipal Corporation",
      location: "Your assigned polling booth",
      importance: "medium",
    },
    {
      id: "pc2024",
      title: "Panchayat Elections",
      date: "December 5, 2024",
      daysLeft: 260,
      type: "Local",
      status: "Upcoming",
      description: "Elections to elect members of the Panchayat",
      location: "Your assigned polling booth",
      importance: "medium",
    },
    {
      id: "se2026",
      title: "State Assembly Elections",
      date: "February 20, 2026",
      daysLeft: 340,
      type: "State",
      status: "Upcoming",
      description: "Elections to elect members of the State Legislative Assembly",
      location: "Your assigned polling booth",
      importance: "high",
    },
  ]

  // Filter elections based on search query and filter type
  const filteredElections = elections.filter((election) => {
    const matchesSearch =
      election.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      election.description.toLowerCase().includes(searchQuery.toLowerCase())

    if (filterType === "all") return matchesSearch
    if (filterType === "national") return matchesSearch && election.type === "National"
    if (filterType === "state") return matchesSearch && election.type === "State"
    if (filterType === "local") return matchesSearch && election.type === "Local"

    return matchesSearch
  })

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Upcoming Elections</h1>
        <p className="text-gray-500">View and prepare for upcoming elections in your area</p>
      </div>

      {/* Election Countdown Banner */}
      <div className="relative mb-6 overflow-hidden rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white shadow-lg">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-white/10 blur-2xl"></div>
        <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-white/10 blur-2xl"></div>

        <div className="relative flex flex-col items-center justify-between gap-4 md:flex-row">
          <div>
            <h2 className="text-2xl font-bold">General Elections 2025</h2>
            <p className="mt-1 text-blue-100">Your vote matters! Be prepared for the upcoming election.</p>
            <div className="mt-4 flex items-center space-x-2">
              <Calendar className="h-5 w-5" />
              <span className="font-medium">April 15, 2025</span>
            </div>
          </div>

          <div className="flex items-center space-x-6">
            <div className="text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/20 text-2xl font-bold backdrop-blur-sm">
                28
              </div>
              <span className="mt-1 text-sm">Days</span>
            </div>

            <Link
              to="/dashboard/elections/ge2025"
              className="rounded-md bg-white px-4 py-2 text-sm font-medium text-blue-700 shadow-sm transition-colors hover:bg-white/90"
            >
              View Details
            </Link>
          </div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="mb-6 flex flex-col space-y-4 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
        <div className="relative w-full sm:max-w-xs">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search elections..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-md border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <span className="text-sm text-gray-500">Filter:</span>
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All Elections</option>
            <option value="national">National</option>
            <option value="state">State</option>
            <option value="local">Local</option>
          </select>
        </div>
      </div>

      {/* Elections List */}
      <div className="grid gap-4 md:grid-cols-2">
        {filteredElections.length > 0 ? (
          filteredElections.map((election) => (
            <Link
              key={election.id}
              to={`/dashboard/elections/${election.id}`}
              className="group overflow-hidden rounded-lg border border-gray-200 bg-white transition-all hover:border-blue-200 hover:shadow-md"
            >
              <div className="border-b bg-gradient-to-r from-blue-50 to-purple-50 px-4 py-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900">{election.title}</h4>
                  <Badge variant={election.importance === "high" ? "warning" : "secondary"}>{election.type}</Badge>
                </div>
              </div>
              <div className="p-4">
                <p className="text-sm text-gray-600">{election.description}</p>

                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <div className="flex items-center space-x-1 text-sm text-gray-500">
                    <Calendar className="h-4 w-4" />
                    <span>{election.date}</span>
                  </div>
                  <div className="flex items-center space-x-1 text-sm text-gray-500">
                    <Clock className="h-4 w-4" />
                    <span>{election.daysLeft} days remaining</span>
                  </div>
                </div>

                <div className="mt-3 flex items-center space-x-1 text-sm text-gray-500">
                  <MapPin className="h-4 w-4" />
                  <span>{election.location}</span>
                </div>

                <Progress value={100 - (election.daysLeft / 365) * 100} max={100} className="mt-3 h-1.5 bg-gray-100" />

                <div className="mt-4 text-right">
                  <span className="inline-flex items-center text-sm font-medium text-blue-600 transition-all group-hover:translate-x-1">
                    View Details
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </span>
                </div>
              </div>
            </Link>
          ))
        ) : (
          <div className="col-span-2 rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
              <Search className="h-6 w-6 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900">No elections found</h3>
            <p className="mt-2 text-sm text-gray-500">
              No elections match your search criteria. Try adjusting your filters.
            </p>
          </div>
        )}
      </div>

      {/* Election Information */}
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Election Information</CardTitle>
          <CardDescription>Important information about the electoral process</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="rounded-lg bg-blue-50 p-4">
              <div className="flex items-start space-x-3">
                <Shield className="mt-0.5 h-5 w-5 text-blue-600" />
                <div>
                  <h4 className="text-sm font-medium text-blue-800">Voter Identification</h4>
                  <p className="mt-1 text-sm text-blue-700">
                    Bring your Voter ID card or any other approved identification document to the polling station. Your
                    QR code will also be required for verification.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-lg bg-blue-50 p-4">
              <div className="flex items-start space-x-3">
                <Clock className="mt-0.5 h-5 w-5 text-blue-600" />
                <div>
                  <h4 className="text-sm font-medium text-blue-800">Voting Hours</h4>
                  <p className="mt-1 text-sm text-blue-700">
                    Polling stations are typically open from 8:00 AM to 6:00 PM. It's recommended to check the exact
                    timing for your polling station before election day.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-lg bg-blue-50 p-4">
              <div className="flex items-start space-x-3">
                <AlertCircle className="mt-0.5 h-5 w-5 text-blue-600" />
                <div>
                  <h4 className="text-sm font-medium text-blue-800">COVID-19 Protocols</h4>
                  <p className="mt-1 text-sm text-blue-700">
                    Follow all COVID-19 safety protocols at the polling station, including wearing a mask and
                    maintaining social distancing.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

