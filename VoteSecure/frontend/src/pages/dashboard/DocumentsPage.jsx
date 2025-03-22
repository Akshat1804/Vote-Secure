"use client"

import React, { useState } from "react"
import { AlertCircle, Download, FileText, Info, Plus, QrCode, Shield, Trash, Upload, X } from "lucide-react"

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

export default function DocumentsPage() {
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)

  // Mock documents data
  const [documents, setDocuments] = useState([
    {
      id: 1,
      name: "Voter ID Card",
      type: "Identity",
      status: "Verified",
      uploadDate: "15 Jan 2024",
      expiryDate: "N/A",
      fileSize: "1.2 MB",
      fileType: "PDF",
      icon: FileText,
    },
    {
      id: 2,
      name: "Aadhaar Card",
      type: "Identity",
      status: "Verified",
      uploadDate: "15 Jan 2024",
      expiryDate: "N/A",
      fileSize: "2.5 MB",
      fileType: "PDF",
      icon: FileText,
    },
    {
      id: 3,
      name: "Address Proof",
      type: "Address",
      status: "Pending",
      uploadDate: "20 Jan 2024",
      expiryDate: "N/A",
      fileSize: "1.8 MB",
      fileType: "PDF",
      icon: FileText,
    },
    {
      id: 4,
      name: "Voter QR Code",
      type: "Voting",
      status: "Active",
      uploadDate: "15 Jan 2024",
      expiryDate: "N/A",
      fileSize: "0.5 MB",
      fileType: "PNG",
      icon: QrCode,
    },
  ])

  // Function to handle document upload
  const handleUpload = () => {
    setIsUploading(true)

    // Simulate upload progress
    let progress = 0
    const interval = setInterval(() => {
      progress += 10
      setUploadProgress(progress)

      if (progress >= 100) {
        clearInterval(interval)

        // Add new document after upload completes
        const newDocument = {
          id: documents.length + 1,
          name: "New Document",
          type: "Other",
          status: "Pending",
          uploadDate: new Date().toLocaleDateString("en-US", { day: "numeric", month: "short", year: "numeric" }),
          expiryDate: "N/A",
          fileSize: "1.5 MB",
          fileType: "PDF",
          icon: FileText,
        }

        setDocuments([...documents, newDocument])
        setIsUploading(false)
        setUploadProgress(0)
        setShowUploadModal(false)
      }
    }, 300)
  }

  // Function to handle document deletion
  const handleDeleteDocument = () => {
    if (documentToDelete) {
      setDocuments(documents.filter((doc) => doc.id !== documentToDelete.id))
      setShowDeleteConfirm(false)
      setDocumentToDelete(null)
    }
  }

  // Function to confirm document deletion
  const confirmDelete = (document) => {
    setDocumentToDelete(document)
    setShowDeleteConfirm(true)
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Documents</h1>
          <p className="text-gray-500">Manage your identification and voting documents</p>
        </div>
        <Button onClick={() => setShowUploadModal(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload Document
        </Button>
      </div>

      {/* Document Categories */}
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <Card className="bg-gradient-to-r from-blue-50 to-blue-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                  <Shield className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Identity Documents</h3>
                  <p className="text-sm text-gray-600">Voter ID, Aadhaar, etc.</p>
                </div>
              </div>
              <Badge variant="success">2 Verified</Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-green-50 to-green-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                  <FileText className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Address Documents</h3>
                  <p className="text-sm text-gray-600">Utility bills, Rent agreement</p>
                </div>
              </div>
              <Badge variant="warning">1 Pending</Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-r from-purple-50 to-purple-100">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                  <QrCode className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Voting Documents</h3>
                  <p className="text-sm text-gray-600">QR Code, Voter slip</p>
                </div>
              </div>
              <Badge variant="success">1 Active</Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Document Guidelines */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <Info className="mt-0.5 h-5 w-5 text-blue-600" />
            <div>
              <h4 className="text-sm font-medium text-blue-800">Document Guidelines</h4>
              <p className="mt-1 text-sm text-blue-700">
                All documents must be clear, legible, and in PDF, JPG, or PNG format. Maximum file size is 5MB. Identity
                and address documents must be valid and not expired. For verification purposes, please ensure all
                information is clearly visible.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Documents List */}
      <Card>
        <CardHeader>
          <CardTitle>My Documents</CardTitle>
          <CardDescription>All your uploaded documents</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {documents.length > 0 ? (
              documents.map((document) => (
                <div key={document.id} className="flex flex-col rounded-lg border sm:flex-row">
                  <div className="flex items-center space-x-4 border-b p-4 sm:border-b-0 sm:border-r sm:p-6">
                    <div
                      className={`flex h-12 w-12 items-center justify-center rounded-lg 
                      ${
                        document.status === "Verified"
                          ? "bg-green-100"
                          : document.status === "Pending"
                            ? "bg-yellow-100"
                            : "bg-blue-100"
                      }`}
                    >
                      <document.icon
                        className={`h-6 w-6 
                        ${
                          document.status === "Verified"
                            ? "text-green-600"
                            : document.status === "Pending"
                              ? "text-yellow-600"
                              : "text-blue-600"
                        }`}
                      />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">{document.name}</h4>
                      <p className="text-sm text-gray-500">{document.type}</p>
                    </div>
                  </div>
                  <div className="flex flex-1 flex-col p-4 sm:p-6">
                    <div className="mb-4 grid grid-cols-2 gap-4 md:grid-cols-4">
                      <div>
                        <p className="text-xs text-gray-500">Status</p>
                        <Badge
                          variant={
                            document.status === "Verified"
                              ? "success"
                              : document.status === "Pending"
                                ? "warning"
                                : "secondary"
                          }
                          className="mt-1"
                        >
                          {document.status}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500">Uploaded On</p>
                        <p className="text-sm font-medium">{document.uploadDate}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500">File Type</p>
                        <p className="text-sm font-medium">{document.fileType}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500">Size</p>
                        <p className="text-sm font-medium">{document.fileSize}</p>
                      </div>
                    </div>
                    <div className="mt-auto flex justify-end space-x-2">
                      <Button variant="outline" size="sm">
                        <Download className="mr-1.5 h-3.5 w-3.5" />
                        Download
                      </Button>
                      <Button variant="destructive" size="sm" onClick={() => confirmDelete(document)}>
                        <Trash className="mr-1.5 h-3.5 w-3.5" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-lg border border-dashed p-8 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
                  <FileText className="h-6 w-6 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">No documents found</h3>
                <p className="mt-2 text-sm text-gray-500">
                  You haven't uploaded any documents yet. Click the button below to upload your first document.
                </p>
                <Button className="mt-4" onClick={() => setShowUploadModal(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Upload Document
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-lg bg-white p-6">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-bold text-gray-900">Upload Document</h3>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 rounded-full p-0"
                onClick={() => {
                  setShowUploadModal(false)
                  setUploadProgress(0)
                  setIsUploading(false)
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {!isUploading ? (
              <>
                <div className="mb-4">
                  <label className="mb-2 block text-sm font-medium text-gray-700">Document Type</label>
                  <select className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500">
                    <option value="">Select document type</option>
                    <option value="identity">Identity Document</option>
                    <option value="address">Address Proof</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div className="mb-4">
                  <label className="mb-2 block text-sm font-medium text-gray-700">Document Name</label>
                  <input
                    type="text"
                    placeholder="Enter document name"
                    className="w-full rounded-md border border-gray-300 p-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="mb-6">
                  <label className="mb-2 block text-sm font-medium text-gray-700">Upload File</label>
                  <div className="rounded-lg border-2 border-dashed border-gray-300 p-6 text-center">
                    <Upload className="mx-auto mb-2 h-8 w-8 text-gray-400" />
                    <p className="text-sm text-gray-600">
                      Drag and drop your file here, or{" "}
                      <span className="text-blue-600 hover:text-blue-800 hover:underline">browse</span>
                    </p>
                    <p className="mt-1 text-xs text-gray-500">Supported formats: PDF, JPG, PNG (Max 5MB)</p>
                  </div>
                </div>

                <div className="rounded-lg bg-yellow-50 p-3 mb-6">
                  <div className="flex items-center">
                    <AlertCircle className="mr-2 h-5 w-5 text-yellow-600" />
                    <span className="text-sm font-medium text-yellow-800">
                      Make sure all information is clearly visible in the document.
                    </span>
                  </div>
                </div>
              </>
            ) : (
              <div className="my-8 space-y-4">
                <div className="text-center">
                  <h4 className="font-medium text-gray-900">Uploading Document...</h4>
                  <p className="text-sm text-gray-500">Please wait while we upload your document.</p>
                </div>

                <div className="relative h-2 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full bg-blue-600 transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>

                <p className="text-center text-sm text-gray-600">{uploadProgress}% Complete</p>
              </div>
            )}

            <div className="flex justify-end space-x-2">
              <Button
                variant="outline"
                onClick={() => {
                  setShowUploadModal(false)
                  setUploadProgress(0)
                  setIsUploading(false)
                }}
                disabled={isUploading}
              >
                Cancel
              </Button>
              <Button onClick={handleUpload} disabled={isUploading}>
                {isUploading ? (
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
                    Uploading...
                  </>
                ) : (
                  "Upload Document"
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-lg bg-white p-6">
            <div className="mb-4 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
                <AlertCircle className="h-6 w-6 text-red-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Confirm Deletion</h3>
              <p className="mt-2 text-gray-600">
                Are you sure you want to delete "{documentToDelete?.name}"? This action cannot be undone.
              </p>
            </div>

            <div className="flex justify-center space-x-4">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteDocument}>
                Delete Document
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

