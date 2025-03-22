"use client"
import { cn } from "../../utils/cn"

const CandidateCard = ({ candidate, isSelected, onSelect, onViewDetails }) => {
  return (
    <div
      className={cn(
        "group relative cursor-pointer overflow-hidden rounded-lg border p-4 transition-all hover:shadow-md",
        isSelected
          ? "border-indigo-500 bg-indigo-50 ring-2 ring-indigo-500 ring-opacity-50"
          : `${candidate.color} hover:border-gray-300`,
      )}
      onClick={() => onSelect(candidate)}
    >
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          <div className="relative h-16 w-16 overflow-hidden rounded-full border-2 border-white shadow-sm transition-transform group-hover:scale-105">
            <img
              src={candidate.image || "/placeholder.svg"}
              alt={candidate.name}
              className="h-full w-full object-cover"
            />
          </div>
          <div className="mt-2 text-center text-2xl">{candidate.symbol}</div>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-medium text-gray-900">{candidate.name}</h3>
          <p className="text-sm text-gray-600">{candidate.party}</p>
          <p className="text-xs text-gray-500">({candidate.partyShort})</p>
          <div className="mt-2 flex space-x-2">
            <button
              type="button"
              className="flex items-center rounded-md bg-white px-2 py-1 text-xs font-medium text-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 transition-colors hover:bg-gray-50"
              onClick={(e) => {
                e.stopPropagation()
                onViewDetails(candidate)
              }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="mr-1 h-3 w-3"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
                <circle cx="12" cy="12" r="3"></circle>
              </svg>
              View Details
            </button>
          </div>
        </div>
        {isSelected && (
          <div className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-indigo-500 text-white">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          </div>
        )}
      </div>
    </div>
  )
}

export default CandidateCard

