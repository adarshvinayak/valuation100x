import { Button } from "@/components/ui/button";
export const Navigation = () => {
  return <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-sm border-b">
      <div className="container mx-auto px-4 py-4 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-primary rounded-lg"></div>
            <h1 className="text-xl font-bold">Valuation100x</h1>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            
            
            
          </div>
          
          
        </div>
      </div>
    </nav>;
};