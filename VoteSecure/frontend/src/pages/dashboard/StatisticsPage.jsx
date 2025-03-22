"use client"

import React, { useState } from "react"
import { BarChart, Calendar, ChevronDown, Download, Info, MapPin, TrendingUp, Users } from "lucide-react"

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

export default function StatisticsPage() {
  const [selectedYear, setSelectedYear] = useState("2024")
  const [selectedRegion, setSelectedRegion] = useState("All")

  // Mock statistics data
  const statisticsData = {
    voterTurnout: {
      national: 67.5,
      state: 72.3,
      local: 58.9,
      trend: [
        { year: "2012", turnout: 58.2 },
        { year: "2016", turnout: 61.8 },
        { year: "2020", turnout: 63.2 },
        { year: "2024", turnout: 67.5 },
      ],
    },
    demographics: {
      ageGroups: [
        { group: "18-25", percentage: 15.2 },
        { group: "26-35", percentage: 22.8 },
        { group: "36-45", percentage: 24.5 },
        { group: "46-60", percentage: 21.3 },
        { group: "60+", percentage: 16.2 },
      ],
      gender: [
        { group: "Male", percentage: 51.2 },
        { group: "Female", percentage: 48.5 },
        { group: "Other", percentage: 0.3 },
      ],
    },
    regions: [
      { name: "North", turnout: 68.7 },
      { name: "South", turnout: 72.1 },
      { name: "East", turnout: 65.3 },
      { name: "West", turnout: 69.8 },
      { name: "Central", turnout: 64.2 },
    ],
    verificationMethods: [
      { method: "Aadhaar + QR", percentage: 72.5 },
      { method: "Voter ID + Biometric", percentage: 25.3 },
      { method: "Other", percentage: 2.2 },
    ],
  }

  // Function to render a simple bar chart
  const renderBarChart = (data, valueKey, labelKey) => {
    const maxValue = Math.max(...data.map((item) => item[valueKey]))

    return (
      <div className="space-y-3">
        {data.map((item, index) => (
          <div key={index}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-gray-600">{item[labelKey]}</span>
              <span className="font-medium text-blue-600">{item[valueKey]}%</span>
            </div>
            <div className="relative h-8 w-full overflow-hidden rounded-md bg-gray-100">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all"
                style={{ width: `${(item[valueKey] / maxValue) * 100}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    )
  }

  // Function to render a simple pie chart
  const renderPieChart = (data, valueKey, labelKey) => {
    const total = data.reduce((sum, item) => sum + item[valueKey], 0)
    let cumulativePercentage = 0

    return (
      <div className="flex items-center justify-center">
        <div className="relative h-48 w-48">
          <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90 transform">
            {data.map((item, index) => {
              const percentage = (item[valueKey] / total) * 100
              const startAngle = cumulativePercentage
              cumulativePercentage += percentage
              const endAngle = cumulativePercentage

              const x1 = 50 + 40 * Math.cos((startAngle / 100) * 2 * Math.PI)
              const y1 = 50 + 40 * Math.sin((startAngle / 100) * 2 * Math.PI)
              const x2 = 50 + 40 * Math.cos((endAngle / 100) * 2 * Math.PI)
              const y2 = 50 + 40 * Math.sin((endAngle / 100) * 2 * Math.PI)

              const largeArcFlag = percentage > 50 ? 1 : 0

              const pathData = [`M 50 50`, `L ${x1} ${y1}`, `A 40 40 0 ${largeArcFlag} 1 ${x2} ${y2}`, `Z`].join(" ")

              const colors = ["fill-blue-500", "fill-purple-500", "fill-green-500", "fill-yellow-500", "fill-red-500"]

              return <path key={index} d={pathData} className={colors[index % colors.length]} />
            })}
            <circle cx="50" cy="50" r="25" fill="white" />
          </svg>
        </div>
        <div className="ml-8 space-y-2">
          {data.map((item, index) => (
            <div key={index} className="flex items-center">
              <div
                className={`mr-2 h-3 w-3 rounded-full ${
                  index === 0
                    ? "bg-blue-500"
                    : index === 1
                      ? "bg-purple-500"
                      : index === 2
                        ? "bg-green-500"
                        : index === 3
                          ? "bg-yellow-500"
                          : "bg-red-500"
                }`}
              ></div>
              <span className="text-sm text-gray-700">
                {item[labelKey]}: {item[valueKey]}%
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Voting Statistics</h1>
        <p className="text-gray-500">Analyze voting trends and participation data</p>
      </div>

      {/* Filters */}
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <div>
          <label htmlFor="year-filter" className="mr-2 text-sm font-medium text-gray-700">
            Year:
          </label>
          <select
            id="year-filter"
            value={selectedYear}
            onChange={(e) => setSelectedYear(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="2024">2024</option>
            <option value="2020">2020</option>
            <option value="2016">2016</option>
            <option value="2012">2012</option>
          </select>
        </div>
        <div>
          <label htmlFor="region-filter" className="mr-2 text-sm font-medium text-gray-700">
            Region:
          </label>
          <select
            id="region-filter"
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="All">All Regions</option>
            <option value="North">North</option>
            <option value="South">South</option>
            <option value="East">East</option>
            <option value="West">West</option>
            <option value="Central">Central</option>
          </select>
        </div>
        <Button variant="outline" size="sm" className="ml-auto">
          <Download className="mr-1.5 h-4 w-4" />
          Export Statistics
        </Button>
      </div>

      {/* Statistics Overview */}
      <div className="mb-6 grid gap-4 md:grid-cols-4">
        <Card className="bg-gradient-to-r from-blue-50 to-blue-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                  <Users className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">National Turnout</p>
                  <h3 className="text-xl font-bold text-gray-900">{statisticsData.voterTurnout.national}%</h3>
                </div>
              </div>
              <TrendingUp className="h-5 w-5 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-green-50 to-green-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                  <MapPin className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">State Turnout</p>
                  <h3 className="text-xl font-bold text-gray-900">{statisticsData.voterTurnout.state}%</h3>
                </div>
              </div>
              <TrendingUp className="h-5 w-5 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-purple-50 to-purple-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                  <MapPin className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Local Turnout</p>
                  <h3 className="text-xl font-bold text-gray-900">{statisticsData.voterTurnout.local}%</h3>
                </div>
              </div>
              <TrendingUp className="h-5 w-5 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-yellow-50 to-yellow-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                  <Calendar className="h-5 w-5 text-yellow-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Election Year</p>
                  <h3 className="text-xl font-bold text-gray-900">{selectedYear}</h3>
                </div>
              </div>
              <ChevronDown className="h-5 w-5 text-gray-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Voter Turnout Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Voter Turnout Trend</CardTitle>
            <CardDescription>Historical voting participation</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 w-full">
              <div className="flex h-full flex-col justify-between">
                <div className="space-y-2">
                  {statisticsData.voterTurnout.trend.map((item, index) => (
                    <div key={index}>
                      <div className="mb-1 flex items-center justify-between text-sm">
                        <span className="text-gray-600">{item.year}</span>
                        <span className="font-medium text-blue-600">{item.turnout}%</span>
                      </div>
                      <div className="relative h-8 w-full overflow-hidden rounded-md bg-gray-100">
                        <div
                          className={`h-full transition-all ${
                            item.year === selectedYear ? "bg-gradient-to-r from-blue-600 to-purple-600" : "bg-blue-400"
                          }`}
                          style={{ width: `${item.turnout}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-lg bg-blue-50 p-3">
                  <div className="flex items-center">
                    <Info className="mr-2 h-5 w-5 text-blue-600" />
                    <span className="text-sm font-medium text-blue-800">
                      Voter turnout has increased by{" "}
                      {(
                        statisticsData.voterTurnout.trend[3].turnout - statisticsData.voterTurnout.trend[0].turnout
                      ).toFixed(1)}
                      % since 2012.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Regional Participation */}
        <Card>
          <CardHeader>
            <CardTitle>Regional Participation</CardTitle>
            <CardDescription>Voter turnout by region</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">{renderBarChart(statisticsData.regions, "turnout", "name")}</div>
          </CardContent>
        </Card>

        {/* Demographic Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Age Demographics</CardTitle>
            <CardDescription>Voter participation by age group</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">{renderPieChart(statisticsData.demographics.ageGroups, "percentage", "group")}</div>
          </CardContent>
        </Card>

        {/* Gender Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Gender Distribution</CardTitle>
            <CardDescription>Voter participation by gender</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">{renderPieChart(statisticsData.demographics.gender, "percentage", "group")}</div>
          </CardContent>
        </Card>
      </div>

      {/* Verification Methods */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Verification Methods</CardTitle>
          <CardDescription>How voters were verified at polling stations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            <div>{renderBarChart(statisticsData.verificationMethods, "percentage", "method")}</div>
            <div className="rounded-lg bg-blue-50 p-4">
              <div className="flex h-full flex-col justify-center">
                <div className="flex items-start space-x-3">
                  <BarChart className="mt-0.5 h-5 w-5 text-blue-600" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800">Verification Insights</h4>
                    <p className="mt-1 text-sm text-blue-700">
                      The majority of voters ({statisticsData.verificationMethods[0].percentage}%) were verified using
                      Aadhaar + QR code method, which is the most secure and efficient verification process. This
                      represents a significant improvement in the voting security compared to previous elections.
                    </p>
                    <p className="mt-2 text-sm text-blue-700">
                      Biometric verification was used as a backup method in cases where QR code verification was not
                      possible.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Sources */}
      <div className="mt-6 rounded-lg border p-4">
        <div className="flex items-start space-x-3">
          <Info className="mt-0.5 h-5 w-5 text-gray-500" />
          <div>
            <h4 className="text-sm font-medium text-gray-700">Data Sources</h4>
            <p className="mt-1 text-sm text-gray-600">
              The statistics presented on this page are sourced from the Election Commission of India and are updated
              after each election cycle. The data is anonymized and aggregated to protect voter privacy.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

