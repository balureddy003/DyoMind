import * as React from "react"
import {
  // AudioWaveform,
  BookOpen,
  Bot,
  // ChartNoAxesCombined,
  // DollarSign,
  Frame,
  Map,
  PieChart,
  Settings2,
  // ShieldAlert,
  // ShoppingBasket,
  SquareTerminal,
  // Wrench,
} from "lucide-react"
import { NavMain } from "@/components/nav-main"
import { NavUser } from "@/components/nav-user"
import { TeamSwitcher } from "@/components/team-switcher"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar"
import { useUserContext } from "@/contexts/UserContext"
import h1 from '@/assets/h1.png'

import { Team, useTeamsContext } from '@/contexts/TeamsContext';

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  onTeamSelect: (team: Team) => void;
  // Removed onUserNameChange prop
}

const data = {
  user: {
    name: "Jon Doe",
    email: "johne@microsoft.com",
    avatar: h1,
  },
  navMain: [
    {
      title: "Playground",
      url: "/",
      icon: SquareTerminal,
      isActive: true,
      items: [
        { title: "Chat", url: "/" },
        { title: "History", url: "/playground-history" },
      ],
    },
    {
      title: "Agent Teams",
      url: "/agents",
      icon: Bot,
      isActive: true,
      items: [{ title: "Library", url: "/agents" }],
    },
    {
      title: "Documentation",
      url: "#",
      icon: BookOpen,
      isActive: true,
      items: [
        {
          title: "Introduction",
          url: "/introduction",
        },
        {
          title: "Get Started",
          url: "/get-started",
        },
      ],
    },
    {
      title: "Settings",
      url: "#",
      icon: Settings2,
      isActive: true,
      items: [
        {
          title: "General",
          url: "/general",
        },
      ],
    },
  ],
  projects: [
    { name: "Design Engineering", url: "#", icon: Frame },
    { name: "Sales & Marketing", url: "#", icon: PieChart },
    { name: "Travel", url: "#", icon: Map },
  ],
}

export function AppSidebar({
  onTeamSelect,
  ...sidebarProps
}: AppSidebarProps) {
  const { userInfo } = useUserContext();
  const { teams } = useTeamsContext();

  return (
    <Sidebar collapsible="icon" {...sidebarProps}>
      <SidebarHeader>
        <TeamSwitcher teams={teams} onTeamSelect={onTeamSelect} />
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={userInfo} />
      </SidebarFooter>
    </Sidebar>
  );
}