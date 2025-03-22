"use client"

import React, { useState } from "react"
import { Clock, Info, MapPin, Navigation, Phone, Search, User } from "lucide-react"

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

export default function PollingMapPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedStation, setSelectedStation] = useState(null)

  // Mock polling stations data
  const pollingStations = [
    {
      id: 1,
      name: "Government High School",
      address: "Plot No. 15, Sector 12, Gandhinagar, Gujarat - 382016",
      boothNumber: "42",
      distance: "1.2 km",
      hours: "8:00 AM - 6:00 PM",
      contactNumber: "+91 9876543210",
      facilities: ["Wheelchair Access", "Drinking Water", "First Aid"],
      assigned: true,
    },
    {
      id: 2,
      name: "Primary School",
      address: "Plot No. 8, Sector 8, Gandhinagar, Gujarat - 382008",
      boothNumber: "15",
      distance: "3.5 km",
      hours: "8:00 AM - 6:00 PM",
      contactNumber: "+91 9876543211",
      facilities: ["Wheelchair Access", "Drinking Water"],
      assigned: false,
    },
    {
      id: 3,
      name: "Community Center",
      address: "Plot No. 22, Sector 16, Gandhinagar, Gujarat - 382016",
      boothNumber: "28",
      distance: "4.8 km",
      hours: "8:00 AM - 6:00 PM",
      contactNumber: "+91 9876543212",
      facilities: ["Wheelchair Access", "Drinking Water", "First Aid", "Restrooms"],
      assigned: false,
    },
  ]

  // Filter polling stations based on search query
  const filteredStations = pollingStations.filter(
    (station) =>
      station.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      station.address.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  // Function to handle station selection
  const handleStationSelect = (station) => {
    setSelectedStation(station)
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Polling Station Map</h1>
        <p className="text-gray-500">Find your assigned polling station and nearby alternatives</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Polling Stations List */}
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Polling Stations</CardTitle>
              <CardDescription>Find stations near you</CardDescription>

              <div className="mt-4 relative">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search stations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full rounded-md border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredStations.length > 0 ? (
                  filteredStations.map((station) => (
                    <div
                      key={station.id}
                      className={`cursor-pointer rounded-lg border p-3 transition-all hover:border-blue-200 hover:bg-blue-50 ${
                        selectedStation?.id === station.id ? "border-blue-500 bg-blue-50" : "border-gray-200"
                      } ${station.assigned ? "border-l-4 border-l-green-500" : ""}`}
                      onClick={() => handleStationSelect(station)}
                    >
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-gray-900">{station.name}</h4>
                        {station.assigned && <Badge variant="success">Assigned</Badge>}
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{station.address}</p>
                      <div className="mt-2 flex items-center justify-between text-sm">
                        <div className="flex items-center text-gray-500">
                          <MapPin className="mr-1 h-3.5 w-3.5" />
                          <span>{station.distance}</span>
                        </div>
                        <div className="flex items-center text-gray-500">
                          <User className="mr-1 h-3.5 w-3.5" />
                          <span>Booth #{station.boothNumber}</span>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-center">
                    <Search className="mx-auto mb-2 h-6 w-6 text-gray-400" />
                    <p className="text-sm text-gray-500">No polling stations found</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Map and Station Details */}
        <div className="md:col-span-2">
          {/* Map */}
          <Card className="mb-6">
            <CardContent className="p-0">
              <div className="aspect-video w-full bg-gray-100 flex items-center justify-center">
                <div className="text-center">
                  <MapPin className="mx-auto mb-2 h-8 w-8 text-gray-400" />
                  <p className="text-sm text-gray-500">Interactive map would be displayed here</p>
                  <p className="text-xs text-gray-400">(Map integration would be implemented in production)</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Station Details */}
          {selectedStation ? (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{selectedStation.name}</CardTitle>
                    <CardDescription>Booth #{selectedStation.boothNumber}</CardDescription>
                  </div>
                  {selectedStation.assigned && <Badge variant="success">Your Assigned Station</Badge>}
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 p-4">
                    <div className="flex items-start space-x-3">
                      <MapPin className="mt-0.5 h-5 w-5 text-blue-600" />
                      <div>
                        <h4 className="text-sm font-medium text-blue-800">Address</h4>
                        <p className="mt-1 text-sm text-blue-700">{selectedStation.address}</p>
                        <div className="mt-3">
                          <Button size="sm" className="bg-blue-600">
                            <Navigation className="mr-1.5 h-3.5 w-3.5" />
                            Get Directions
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-lg border border-gray-200 p-4">
                      <div className="flex items-start space-x-3">
                        <Clock className="mt-0.5 h-5 w-5 text-gray-500" />
                        <div>
                          <h4 className="text-sm font-medium text-gray-700">Voting Hours</h4>
                          <p className="mt-1 text-sm text-gray-600">{selectedStation.hours}</p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-lg border border-gray-200 p-4">
                      <div className="flex items-start space-x-3">
                        <Phone className="mt-0.5 h-5 w-5 text-gray-500" />
                        <div>
                          <h4 className="text-sm font-medium text-gray-700">Contact Number</h4>
                          <p className="mt-1 text-sm text-gray-600">{selectedStation.contactNumber}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="mb-2 text-sm font-medium text-gray-700">Available Facilities</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedStation.facilities.map((facility, index) => (
                        <Badge key={index} variant="secondary">
                          {facility}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-lg bg-blue-50 p-4">
                    <div className="flex items-start space-x-3">
                      <Info className="mt-0.5 h-5 w-5 text-blue-600" />
                      <div>
                        <h4 className="text-sm font-medium text-blue-800">Important Information</h4>
                        <p className="mt-1 text-sm text-blue-700">
                          Bring your Voter ID card and QR code for verification. Arrive early to avoid long queues.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-6">
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <MapPin className="mb-4 h-12 w-12 text-gray-300" />
                  <h3 className="text-lg font-medium text-gray-900">Select a Polling Station</h3>
                  <p className="mt-2 text-gray-500">
                    Select a polling station from the list to view its details and location on the map.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

