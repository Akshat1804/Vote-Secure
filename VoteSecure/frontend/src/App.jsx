"use client"

import { useEffect } from "react"
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import HomePage from "./pages/HomePage"
import AadhaarVerificationPage from "./pages/AadhaarVerificationPage"
import OtpVerificationPage from "./pages/OtpVerificationPage"
import SuccessPage from "./pages/SuccessPage"
import NotFoundPage from "./pages/NotFoundPage"
import DashboardLayout from "./layouts/DashboardLayout"
import DashboardHome from "./pages/dashboard/DashboardHome"
import ProfilePage from "./pages/dashboard/ProfilePage"
import ElectionsPage from "./pages/dashboard/ElectionsPage"
import ElectionDetailsPage from "./pages/dashboard/ElectionDetailsPage"
import DocumentsPage from "./pages/dashboard/DocumentsPage"
import SettingsPage from "./pages/dashboard/SettingsPage"
import HelpSupportPage from "./pages/dashboard/HelpSupportPage"
import PollingMapPage from "./pages/dashboard/PollingMapPage"
import VoteVerificationPage from "./pages/voting/VoteVerificationPage"
import VotingPage from "./pages/voting/VotingPage"
import AdminDashboard from "./pages/admin/AdminDashboard"
import AdminHome from "./pages/admin/AdminHome"
import VoterManagement from "./pages/admin/VoterManagement"
import RegisterUserPage from "./pages/admin/RegisterUserPage"
import { VerificationProvider } from "./context/VerificationContext"
import { VoterProvider } from "./context/VoterContext"
import { initializeUserVotingStatus } from "./utils/voterStatusUtils"

function App() {
  // Initialize user voting status when the app loads
  useEffect(() => {
    initializeUserVotingStatus()
  }, [])

  return (
    <VoterProvider>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<HomePage />} />
          <Route path="/verify-aadhaar" element={<AadhaarVerificationPage />} />
          <Route path="/verify-otp" element={<OtpVerificationPage />} />
          <Route path="/success" element={<SuccessPage />} />

          {/* Dashboard routes */}
          <Route path="/dashboard" element={<DashboardLayout />}>
            <Route index element={<DashboardHome />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="elections" element={<ElectionsPage />} />
            <Route path="elections/:id" element={<ElectionDetailsPage />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="help" element={<HelpSupportPage />} />
            <Route path="polling-map" element={<PollingMapPage />} />
          </Route>

          {/* Voting routes */}
          <Route
            path="/voting/verify"
            element={
              <VerificationProvider>
                <VoteVerificationPage />
              </VerificationProvider>
            }
          />
          <Route path="/voting/cast-vote" element={<VotingPage />} />

          {/* Admin routes */}
          <Route path="/admin" element={<AdminDashboard />}>
            <Route index element={<AdminHome />} />
            <Route path="voters" element={<VoterManagement />} />
            <Route path="register-user" element={<RegisterUserPage />} />
          </Route>

          {/* Fallback routes */}
          <Route path="/404" element={<NotFoundPage />} />
          <Route path="*" element={<Navigate to="/404" replace />} />
        </Routes>
      </Router>
    </VoterProvider>
  )
}

export default App

