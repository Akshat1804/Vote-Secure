"use client"

import React from "react"
import { Link } from "react-router-dom"
import {
  ChevronRight,
  Plus,
  User,
  Users,
  CheckCircle,
  AlertTriangle,
  Clock,
  Download,
  FileText,
  BarChart,
} from "lucide-react"
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

    if (asChild) {
      return <Comp ref={ref} {...props} />
    }

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

export default function AdminHome() {
  const { stats, getRecentRegistrations } = useVoters()
  const recentRegistrations = getRecentRegistrations(5)

  // Calculate verification rate
  const verificationRate = stats.totalVoters > 0 ? Math.round((stats.verifiedVoters / stats.totalVoters) * 100) : 0

  return (
    <div className="p-4 md:p-6">
      {/* Welcome Banner */}
      <div className="relative mb-6 overflow-hidden rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white shadow-lg">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-white/10 blur-2xl"></div>
        <div className="absolute -bottom-10 -left-10 h-40 w-40 rounded-full bg-white/10 blur-2xl"></div>

        <div className="relative">
          <h1 className="text-2xl font-bold">Voter Registration Admin</h1>
          <p className="mt-2 text-blue-100">Register and manage voter data from a central location.</p>

          <div className="mt-4 flex flex-wrap gap-3">
            <Link
              to="/admin/register-user"
              className="inline-flex items-center rounded-full bg-white/20 px-4 py-1 text-sm font-medium text-white backdrop-blur-sm transition-colors hover:bg-white/30"
            >
              Register New Voter
              <ChevronRight className="ml-1 h-4 w-4" />
            </Link>
            <Link
              to="/admin/voters"
              className="inline-flex items-center rounded-full bg-white/20 px-4 py-1 text-sm font-medium text-white backdrop-blur-sm transition-colors hover:bg-white/30"
            >
              Manage Voters
              <ChevronRight className="ml-1 h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-white">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Total Voters</p>
                <h3 className="text-2xl font-bold text-gray-900">{stats.totalVoters.toLocaleString()}</h3>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">New Registrations</p>
                <h3 className="text-2xl font-bold text-gray-900">{stats.newRegistrations.toLocaleString()}</h3>
                <p className="text-xs text-gray-500 mt-1">Today</p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                <User className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Pending Verifications</p>
                <h3 className="text-2xl font-bold text-gray-900">{stats.pendingVerifications.toLocaleString()}</h3>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-yellow-100">
                <Clock className="h-6 w-6 text-yellow-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Stats */}
      <div className="grid gap-4 md:grid-cols-2 mt-6">
        <Card className="bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Verification Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="mr-2 flex h-8 w-8 items-center justify-center rounded-full bg-green-100">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Verified</p>
                    <p className="text-xs text-gray-500">{stats.verifiedVoters} voters</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">
                    {stats.totalVoters > 0 ? Math.round((stats.verifiedVoters / stats.totalVoters) * 100) : 0}%
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="mr-2 flex h-8 w-8 items-center justify-center rounded-full bg-yellow-100">
                    <Clock className="h-4 w-4 text-yellow-600" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Pending</p>
                    <p className="text-xs text-gray-500">{stats.pendingVerifications} voters</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">
                    {stats.totalVoters > 0 ? Math.round((stats.pendingVerifications / stats.totalVoters) * 100) : 0}%
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="mr-2 flex h-8 w-8 items-center justify-center rounded-full bg-red-100">
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Rejected</p>
                    <p className="text-xs text-gray-500">{stats.rejectedVoters} voters</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">
                    {stats.totalVoters > 0 ? Math.round((stats.rejectedVoters / stats.totalVoters) * 100) : 0}%
                  </p>
                </div>
              </div>

              {/* Simple bar chart visualization */}
              <div className="mt-4">
                <div className="flex h-4 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full bg-green-500"
                    style={{
                      width: `${stats.totalVoters > 0 ? (stats.verifiedVoters / stats.totalVoters) * 100 : 0}%`,
                    }}
                  ></div>
                  <div
                    className="h-full bg-yellow-500"
                    style={{
                      width: `${stats.totalVoters > 0 ? (stats.pendingVerifications / stats.totalVoters) * 100 : 0}%`,
                    }}
                  ></div>
                  <div
                    className="h-full bg-red-500"
                    style={{
                      width: `${stats.totalVoters > 0 ? (stats.rejectedVoters / stats.totalVoters) * 100 : 0}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Registration Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">Verification Rate</p>
                <p className="text-sm font-medium">{verificationRate}%</p>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200">
                <div className="h-full rounded-full bg-blue-600" style={{ width: `${verificationRate}%` }}></div>
              </div>

              <div className="mt-4 rounded-lg bg-blue-50 p-4">
                <div className="flex items-start space-x-3">
                  <BarChart className="h-5 w-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800">Registration Summary</h4>
                    <p className="mt-1 text-sm text-blue-700">
                      {stats.totalVoters} total voters registered, with {stats.verifiedVoters} verified and{" "}
                      {stats.pendingVerifications} pending verification.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-between mt-4">
                <Button variant="outline" size="sm" className="flex items-center gap-1">
                  <Download className="h-4 w-4" />
                  Export Report
                </Button>
                <Button variant="outline" size="sm" className="flex items-center gap-1">
                  <FileText className="h-4 w-4" />
                  Generate PDF
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Registrations */}
      <Card className="mt-6 bg-white">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Recent Registrations</CardTitle>
            <Link to="/admin/voters" className="text-sm font-medium text-blue-600 hover:underline">
              View All
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b text-left text-sm font-medium text-gray-500">
                    <th className="pb-2 pl-0">Voter ID</th>
                    <th className="pb-2">Name</th>
                    <th className="pb-2">Date</th>
                    <th className="pb-2 pr-0">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {recentRegistrations.length > 0 ? (
                    recentRegistrations.map((registration) => (
                      <tr key={registration.id} className="text-sm">
                        <td className="py-3 pl-0 font-medium">{registration.id}</td>
                        <td className="py-3">{registration.name}</td>
                        <td className="py-3">{registration.registrationDate}</td>
                        <td className="py-3 pr-0">
                          <span
                            className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${
                              registration.status === "Verified"
                                ? "bg-green-100 text-green-800"
                                : registration.status === "Pending"
                                  ? "bg-yellow-100 text-yellow-800"
                                  : "bg-red-100 text-red-800"
                            }`}
                          >
                            {registration.status}
                          </span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="py-6 text-center">
                        <p className="text-gray-500">No registrations yet</p>
                        <p className="text-sm text-gray-400">Register new voters to see them here</p>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <Button asChild variant="outline" className="w-full">
              <Link to="/admin/register-user">
                <Plus className="mr-2 h-4 w-4" />
                Register New Voter
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="mt-6">
        <h2 className="mb-4 text-lg font-bold text-gray-900">Quick Actions</h2>
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          <Link
            to="/admin/register-user"
            className="flex items-center space-x-3 rounded-lg border border-gray-200 bg-white p-4 transition-all hover:border-blue-200 hover:shadow-md"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
              <User className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Register Voter</h3>
              <p className="text-sm text-gray-500">Add new voter</p>
            </div>
          </Link>

          <Link
            to="/admin/voters"
            className="flex items-center space-x-3 rounded-lg border border-gray-200 bg-white p-4 transition-all hover:border-blue-200 hover:shadow-md"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
              <Users className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Manage Voters</h3>
              <p className="text-sm text-gray-500">View and edit</p>
            </div>
          </Link>

          <div className="flex items-center space-x-3 rounded-lg border border-gray-200 bg-white p-4 transition-all hover:border-blue-200 hover:shadow-md">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
              <CheckCircle className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Verify Voters</h3>
              <p className="text-sm text-gray-500">Process pending</p>
            </div>
          </div>

          <div className="flex items-center space-x-3 rounded-lg border border-gray-200 bg-white p-4 transition-all hover:border-blue-200 hover:shadow-md">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
              <Download className="h-5 w-5 text-yellow-600" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Export Data</h3>
              <p className="text-sm text-gray-500">Download reports</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

