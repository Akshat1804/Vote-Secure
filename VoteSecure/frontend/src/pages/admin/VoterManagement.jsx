"use client"

import React, { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import {
  ArrowUpDown,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  Edit,
  Filter,
  MoreHorizontal,
  Plus,
  Search,
  Trash,
  User,
  X,
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

// Add Avatar component after Badge component
const Avatar = React.forwardRef(({ className, src, alt, fallback, ...props }, ref) => {
  const [error, setError] = useState(false)

  return (
    <div
      ref={ref}
      className={cn("relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full", className)}
      {...props}
    >
      {!error && src ? (
        <img
          className="aspect-square h-full w-full object-cover"
          src={src || "/placeholder.svg"}
          alt={alt}
          onError={() => setError(true)}
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center rounded-full bg-gray-200 text-gray-600">
          {fallback || alt?.charAt(0)?.toUpperCase() || "U"}
        </div>
      )}
    </div>
  )
})
Avatar.displayName = "Avatar"

export default function VoterManagement() {
  const { voters, deleteVoter, changeVoterStatus } = useVoters()
  const [searchQuery, setSearchQuery] = useState("")
  const [filterStatus, setFilterStatus] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedVoters, setSelectedVoters] = useState([])
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [voterToDelete, setVoterToDelete] = useState(null)
  const [showVoterDetails, setShowVoterDetails] = useState(false)
  const [selectedVoterDetails, setSelectedVoterDetails] = useState(null)
  const [sortField, setSortField] = useState("registrationDate")
  const [sortDirection, setSortDirection] = useState("desc")
  const [viewMode, setViewMode] = useState("table") // "table" or "card"

  // Filter voters based on search query and status
  const filteredVoters = voters.filter((voter) => {
    const matchesSearch =
      voter.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      voter.id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (voter.aadhaar && voter.aadhaar.replace(/\s/g, "").includes(searchQuery.replace(/\s/g, "")))

    if (filterStatus === "all") return matchesSearch
    return matchesSearch && voter.status?.toLowerCase() === filterStatus.toLowerCase()
  })

  // Sort voters
  const sortedVoters = [...filteredVoters].sort((a, b) => {
    if (sortField === "registrationDate") {
      const dateA = new Date(a.registrationDate)
      const dateB = new Date(b.registrationDate)
      return sortDirection === "asc" ? dateA - dateB : dateB - dateA
    } else if (sortField === "name") {
      return sortDirection === "asc" ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name)
    } else if (sortField === "id") {
      return sortDirection === "asc" ? a.id.localeCompare(b.id) : b.id.localeCompare(a.id)
    }
    return 0
  })

  // Pagination
  const itemsPerPage = viewMode === "table" ? 5 : 6
  const totalPages = Math.ceil(sortedVoters.length / itemsPerPage)
  const paginatedVoters = sortedVoters.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

  // Reset page when filter changes
  useEffect(() => {
    setCurrentPage(1)
  }, [filterStatus, searchQuery])

  // Handle sort
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  // Handle select all
  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedVoters(paginatedVoters.map((voter) => voter.id))
    } else {
      setSelectedVoters([])
    }
  }

  // Handle select voter
  const handleSelectVoter = (voterId) => {
    if (selectedVoters.includes(voterId)) {
      setSelectedVoters(selectedVoters.filter((id) => id !== voterId))
    } else {
      setSelectedVoters([...selectedVoters, voterId])
    }
  }

  // Handle delete confirmation
  const confirmDelete = (voter) => {
    setVoterToDelete(voter)
    setShowDeleteConfirm(true)
  }

  // Handle delete voter
  const handleDeleteVoter = () => {
    if (voterToDelete) {
      deleteVoter(voterToDelete.id)
      setShowDeleteConfirm(false)
      setVoterToDelete(null)
    }
  }

  // Handle view voter details
  const viewVoterDetails = (voter) => {
    setSelectedVoterDetails(voter)
    setShowVoterDetails(true)
  }

  // Handle verify voter
  const handleVerifyVoter = (id) => {
    changeVoterStatus(id, "Verified")
    if (selectedVoterDetails && selectedVoterDetails.id === id) {
      setSelectedVoterDetails({ ...selectedVoterDetails, status: "Verified" })
    }
  }

  // Handle reject voter
  const handleRejectVoter = (id) => {
    changeVoterStatus(id, "Rejected")
    if (selectedVoterDetails && selectedVoterDetails.id === id) {
      setSelectedVoterDetails({ ...selectedVoterDetails, status: "Rejected" })
    }
  }

  // Export voters data
  const exportVotersData = () => {
    const dataToExport = filteredVoters.map((voter) => ({
      ID: voter.id,
      Name: voter.name,
      Aadhaar: voter.aadhaar,
      DOB: voter.dob,
      Gender: voter.gender,
      Address: voter.address,
      Phone: voter.phone,
      Email: voter.email || "N/A",
      Constituency: voter.constituency,
      "Polling Station": voter.pollingStation,
      "Registration Date": voter.registrationDate,
      Status: voter.status,
    }))

    const csvContent =
      "data:text/csv;charset=utf-8," +
      Object.keys(dataToExport[0]).join(",") +
      "\n" +
      dataToExport
        .map((row) => {
          return Object.values(row)
            .map((value) => {
              // Wrap values with commas in quotes
              return typeof value === "string" && value.includes(",") ? `"${value}"` : value
            })
            .join(",")
        })
        .join("\n")

    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `voters_data_${new Date().toISOString().split("T")[0]}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Toggle view mode between table and card
  const toggleViewMode = () => {
    setViewMode(viewMode === "table" ? "card" : "table")
    setCurrentPage(1) // Reset to first page when changing view
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Voter Management</h1>
        <p className="text-gray-500">View and manage registered voters</p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col space-y-4 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
            <div className="flex flex-1 items-center space-x-2">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search voters..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full rounded-md border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-gray-500" />
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="rounded-md border border-gray-200 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="verified">Verified</option>
                  <option value="pending">Pending</option>
                  <option value="rejected">Rejected</option>
                </select>
              </div>
            </div>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" onClick={toggleViewMode}>
                {viewMode === "table" ? "Card View" : "Table View"}
              </Button>
              <Button variant="outline" size="sm" onClick={exportVotersData}>
                <Download className="mr-2 h-4 w-4" />
                Export
              </Button>
              <Button asChild size="sm">
                <Link to="/admin/register-user">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Voter
                </Link>
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Table View */}
          {viewMode === "table" && (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b text-left text-sm font-medium text-gray-500">
                    <th className="px-4 py-3 pl-0">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          onChange={handleSelectAll}
                          checked={selectedVoters.length === paginatedVoters.length && paginatedVoters.length > 0}
                        />
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center">
                        <span>Photo</span>
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center cursor-pointer" onClick={() => handleSort("id")}>
                        <span>Voter ID</span>
                        <ArrowUpDown className={`ml-1 h-4 w-4 ${sortField === "id" ? "text-blue-600" : ""}`} />
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center cursor-pointer" onClick={() => handleSort("name")}>
                        <span>Name</span>
                        <ArrowUpDown className={`ml-1 h-4 w-4 ${sortField === "name" ? "text-blue-600" : ""}`} />
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center">
                        <span>Constituency</span>
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center cursor-pointer" onClick={() => handleSort("registrationDate")}>
                        <span>Registration Date</span>
                        <ArrowUpDown
                          className={`ml-1 h-4 w-4 ${sortField === "registrationDate" ? "text-blue-600" : ""}`}
                        />
                      </div>
                    </th>
                    <th className="px-4 py-3">
                      <div className="flex items-center">
                        <span>Status</span>
                      </div>
                    </th>
                    <th className="px-4 py-3 pr-0 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {paginatedVoters.length > 0 ? (
                    paginatedVoters.map((voter) => (
                      <tr key={voter.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3 pl-0">
                          <div className="flex items-center">
                            <input
                              type="checkbox"
                              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                              checked={selectedVoters.includes(voter.id)}
                              onChange={() => handleSelectVoter(voter.id)}
                            />
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <Avatar src={voter.avatar} alt={voter.name} fallback={voter.name.charAt(0)} />
                        </td>
                        <td className="px-4 py-3 font-medium">{voter.id}</td>
                        <td className="px-4 py-3">{voter.name}</td>
                        <td className="px-4 py-3">{voter.constituency}</td>
                        <td className="px-4 py-3">{voter.registrationDate}</td>
                        <td className="px-4 py-3">
                          <Badge
                            variant={
                              voter.status === "Verified"
                                ? "success"
                                : voter.status === "Pending"
                                  ? "warning"
                                  : "destructive"
                            }
                          >
                            {voter.status}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 pr-0 text-right">
                          <div className="flex items-center justify-end space-x-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0"
                              onClick={() => viewVoterDetails(voter)}
                            >
                              <span className="sr-only">View details</span>
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0 text-red-500 hover:bg-red-50 hover:text-red-600"
                              onClick={() => confirmDelete(voter)}
                            >
                              <span className="sr-only">Delete</span>
                              <Trash className="h-4 w-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8} className="px-4 py-6 text-center">
                        <div className="flex flex-col items-center">
                          <User className="h-10 w-10 text-gray-400" />
                          <p className="mt-2 text-gray-500">No voters registered yet</p>
                          <p className="text-sm text-gray-400">Use the "Add Voter" button to register new voters</p>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Card View */}
          {viewMode === "card" && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {paginatedVoters.length > 0 ? (
                paginatedVoters.map((voter) => (
                  <div
                    key={voter.id}
                    className="rounded-lg border border-gray-200 bg-white overflow-hidden hover:shadow-lg transition-all"
                  >
                    <div className="p-6 flex flex-col items-center">
                      <div className="relative mb-4">
                        {voter.avatar ? (
                          <img
                            src={voter.avatar || "/placeholder.svg"}
                            alt={voter.name}
                            className="h-36 w-36 rounded-full object-cover border-4 border-gray-100 shadow-md"
                            onError={(e) => {
                              e.target.onerror = null
                              e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(voter.name)}&background=random&color=fff&size=128`
                            }}
                          />
                        ) : (
                          <div className="h-36 w-36 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center text-gray-600 text-5xl font-medium border-4 border-gray-100 shadow-md">
                            {voter.name.charAt(0)}
                          </div>
                        )}
                        <Badge
                          variant={
                            voter.status === "Verified"
                              ? "success"
                              : voter.status === "Pending"
                                ? "warning"
                                : "destructive"
                          }
                          className="absolute bottom-0 right-0 transform translate-x-1/4 px-3 py-1 text-sm"
                        >
                          {voter.status}
                        </Badge>
                      </div>
                      <h3 className="font-bold text-xl text-gray-900 text-center">{voter.name}</h3>
                      <p className="text-sm text-gray-500 text-center mt-1 bg-gray-100 px-3 py-1 rounded-full">
                        {voter.id}
                      </p>

                      <div className="w-full mt-4 pt-4 border-t border-gray-100">
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Constituency</span>
                            <p className="truncate font-medium text-gray-800">{voter.constituency}</p>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Registered</span>
                            <p className="font-medium text-gray-800">{voter.registrationDate}</p>
                          </div>
                          <div className="col-span-2">
                            <span className="text-xs font-medium text-gray-500 block">Polling Station</span>
                            <p className="truncate font-medium text-gray-800">{voter.pollingStation}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="bg-gray-50 px-6 py-4 flex justify-between items-center border-t border-gray-100">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-blue-600 hover:text-blue-800 hover:bg-blue-50 font-medium"
                        onClick={() => viewVoterDetails(voter)}
                      >
                        View Details
                      </Button>
                      <div className="flex space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-9 w-9 p-0 text-green-600 hover:bg-green-50 hover:text-green-700 rounded-full"
                          onClick={() => handleVerifyVoter(voter.id)}
                          disabled={voter.status === "Verified"}
                        >
                          <Check className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-9 w-9 p-0 text-red-500 hover:bg-red-50 hover:text-red-600 rounded-full"
                          onClick={() => confirmDelete(voter)}
                        >
                          <Trash className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-full p-12 text-center border rounded-lg bg-gray-50">
                  <div className="flex flex-col items-center">
                    <User className="h-20 w-20 text-gray-300 mb-4" />
                    <h3 className="text-xl font-medium text-gray-900 mb-2">No voters registered yet</h3>
                    <p className="text-gray-500 mb-6 max-w-md mx-auto">
                      Use the "Add Voter" button to register new voters and they will appear here in card format with
                      their photos.
                    </p>
                    <Button asChild size="lg">
                      <Link to="/admin/register-user">
                        <Plus className="mr-2 h-5 w-5" />
                        Add Voter
                      </Link>
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Pagination */}
          {filteredVoters.length > 0 && (
            <div className="mt-4 flex items-center justify-between">
              <p className="text-sm text-gray-500">
                Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                {Math.min(currentPage * itemsPerPage, filteredVoters.length)} of {filteredVoters.length} voters
              </p>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm">
                  Page {currentPage} of {totalPages || 1}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={currentPage === totalPages || totalPages === 0}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-lg bg-white p-6">
            <div className="mb-4 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
                <Trash className="h-6 w-6 text-red-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Confirm Deletion</h3>
              <p className="mt-2 text-gray-600">
                Are you sure you want to delete voter "{voterToDelete?.name}" with ID {voterToDelete?.id}? This action
                cannot be undone.
              </p>
            </div>

            <div className="flex justify-center space-x-4">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteVoter}>
                Delete Voter
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Voter Details Modal */}
      {showVoterDetails && selectedVoterDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 overflow-y-auto">
          <div className="w-full max-w-3xl rounded-lg bg-white shadow-xl">
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <h3 className="text-xl font-bold text-gray-900">Voter Details</h3>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 rounded-full"
                onClick={() => setShowVoterDetails(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="p-6">
              <div className="flex flex-col md:flex-row gap-6">
                {/* Left column with photo */}
                <div className="md:w-1/3 flex flex-col items-center">
                  {selectedVoterDetails.avatar ? (
                    <img
                      src={selectedVoterDetails.avatar || "/placeholder.svg"}
                      alt={selectedVoterDetails.name}
                      className="h-48 w-48 rounded-full object-cover border-4 border-gray-100 shadow-lg"
                      onError={(e) => {
                        e.target.onerror = null
                        e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(selectedVoterDetails.name)}&background=random&color=fff&size=128`
                      }}
                    />
                  ) : (
                    <div className="h-48 w-48 rounded-full bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center text-gray-600 text-6xl font-medium border-4 border-gray-100 shadow-lg">
                      {selectedVoterDetails.name.charAt(0)}
                    </div>
                  )}

                  <h2 className="mt-4 text-xl font-bold text-center">{selectedVoterDetails.name}</h2>
                  <div className="mt-2 px-3 py-1 bg-blue-100 rounded-full text-blue-800 text-sm font-medium">
                    {selectedVoterDetails.id}
                  </div>

                  <div className="mt-4 w-full">
                    <Badge
                      variant={
                        selectedVoterDetails.status === "Verified"
                          ? "success"
                          : selectedVoterDetails.status === "Pending"
                            ? "warning"
                            : "destructive"
                      }
                      className="w-full flex justify-center py-1 text-sm"
                    >
                      {selectedVoterDetails.status}
                    </Badge>
                  </div>
                </div>

                {/* Right column with details */}
                <div className="md:w-2/3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 mb-1">Personal Information</h4>
                        <div className="space-y-2 rounded-lg border p-3 bg-gray-50">
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Date of Birth</span>
                            <span className="text-sm font-medium text-gray-900">{selectedVoterDetails.dob}</span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Gender</span>
                            <span className="text-sm font-medium text-gray-900">{selectedVoterDetails.gender}</span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Aadhaar</span>
                            <span className="text-sm font-medium text-gray-900">{selectedVoterDetails.aadhaar}</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-gray-500 mb-1">Contact Information</h4>
                        <div className="space-y-2 rounded-lg border p-3 bg-gray-50">
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Phone</span>
                            <span className="text-sm font-medium text-gray-900">{selectedVoterDetails.phone}</span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Email</span>
                            <span className="text-sm font-medium text-gray-900">
                              {selectedVoterDetails.email || "Not provided"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-500 mb-1">Voting Information</h4>
                        <div className="space-y-2 rounded-lg border p-3 bg-gray-50">
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Constituency</span>
                            <span className="text-sm font-medium text-gray-900">
                              {selectedVoterDetails.constituency}
                            </span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Polling Station</span>
                            <span className="text-sm font-medium text-gray-900">
                              {selectedVoterDetails.pollingStation}
                            </span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-gray-500 block">Registration Date</span>
                            <span className="text-sm font-medium text-gray-900">
                              {selectedVoterDetails.registrationDate}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-gray-500 mb-1">Address</h4>
                        <div className="rounded-lg border p-3 bg-gray-50">
                          <p className="text-sm text-gray-900">{selectedVoterDetails.address}</p>
                        </div>
                      </div>
                    </div>

                    <div className="md:col-span-2">
                      <h4 className="text-sm font-medium text-gray-500 mb-1">Actions</h4>
                      <div className="grid grid-cols-4 gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => handleVerifyVoter(selectedVoterDetails.id)}
                          disabled={selectedVoterDetails.status === "Verified"}
                        >
                          <Check className="mr-2 h-4 w-4" />
                          Verify
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => handleRejectVoter(selectedVoterDetails.id)}
                          disabled={selectedVoterDetails.status === "Rejected"}
                        >
                          <X className="mr-2 h-4 w-4" />
                          Reject
                        </Button>
                        <Button variant="outline" size="sm" className="w-full">
                          <Edit className="mr-2 h-4 w-4" />
                          Edit
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          className="w-full"
                          onClick={() => {
                            setShowVoterDetails(false)
                            confirmDelete(selectedVoterDetails)
                          }}
                        >
                          <Trash className="mr-2 h-4 w-4" />
                          Delete
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end p-6 border-t border-gray-200 bg-gray-50">
              <Button onClick={() => setShowVoterDetails(false)}>Close</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

