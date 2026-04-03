```python
# solution.py

"""
VideoCollaborationSuite - A collaborative video editing application
Supports multiple users, subtitle synchronization, playback speed control,
real-time chat, and version control for video projects.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import copy
import uuid


class UserStatus(Enum):
    """Enum for user connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


class EditType(Enum):
    """Types of edits that can be made to a video project"""
    TRIM = "trim"
    SUBTITLE_SYNC = "subtitle_sync"
    PLAYBACK_SPEED = "playback_speed"
    COMMENT = "comment"
    VERSION_SAVE = "version_save"


@dataclass
class User:
    """Represents a user in the collaboration system"""
    user_id: str
    username: str
    email: str
    status: UserStatus = UserStatus.OFFLINE
    joined_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    def set_online(self):
        """Set user status to online"""
        self.status = UserStatus.ONLINE
        self.joined_at = datetime.now()
        self.last_active = datetime.now()
    
    def set_offline(self):
        """Set user status to offline"""
        self.status = UserStatus.OFFLINE
        self.last_active = datetime.now()
    
    def set_busy(self):
        """Set user status to busy"""
        self.status = UserStatus.BUSY
        self.last_active = datetime.now()


@dataclass
class Subtitle:
    """Represents a subtitle entry with timing information"""
    subtitle_id: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    text: str
    language: str = "en"
    is_manual: bool = False  # True if manually adjusted
    
    def to_dict(self) -> Dict:
        """Convert subtitle to dictionary"""
        return {
            'subtitle_id': self.subtitle_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text,
            'language': self.language,
            'is_manual': self.is_manual
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Subtitle':
        """Create subtitle from dictionary"""
        return cls(
            subtitle_id=data['subtitle_id'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            text=data['text'],
            language=data.get('language', 'en'),
            is_manual=data.get('is_manual', False)
        )


@dataclass
class ChatMessage:
    """Represents a chat message in the collaboration system"""
    message_id: str
    user_id: str
    username: str
    content: str
    timestamp: datetime
    message_type: str = "text"  # text, suggestion, feedback
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            'message_id': self.message_id,
            'user_id': self.user_id,
            'username': self.username,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            user_id=data['user_id'],
            username=data['username'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            message_type=data.get('message_type', 'text')
        )


@dataclass
class VideoVersion:
    """Represents a version of the video project"""
    version_id: str
    version_number: int
    created_by: str
    created_at: datetime
    description: str
    video_state: Dict  # Snapshot of video state at this version
    is_current: bool = False
    
    def to_dict(self) -> Dict:
        """Convert version to dictionary"""
        return {
            'version_id': self.version_id,
            'version_number': self.version_number,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'video_state': self.video_state,
            'is_current': self.is_current
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoVersion':
        """Create version from dictionary"""
        return cls(
            version_id=data['version_id'],
            version_number=data['version_number'],
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            description=data['description'],
            video_state=data['video_state'],
            is_current=data.get('is_current', False)
        )


@dataclass
class VideoProject:
    """Represents a video project with all its properties"""
    project_id: str
    title: str
    description: str
    video_file: str
    duration: float  # in seconds
    playback_speed: float = 1.0
    subtitles: List[Subtitle] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    active_users: List[str] = field(default_factory=list)
    versions: List[VideoVersion] = field(default_factory=list)
    current_version: int = 1
    
    def add_subtitle(self, subtitle: Subtitle):
        """Add a subtitle to the project"""
        self.subtitles.append(subtitle)
        self.last_modified = datetime.now()
    
    def remove_subtitle(self, subtitle_id: str):
        """Remove a subtitle by ID"""
        self.subtitles = [s for s in self.subtitles if s.subtitle_id != subtitle_id]
        self.last_modified = datetime.now()
    
    def update_subtitle(self, subtitle_id: str, new_subtitle: Subtitle):
        """Update an existing subtitle"""
        for i, s in enumerate(self.subtitles):
            if s.subtitle_id == subtitle_id:
                self.subtitles[i] = new_subtitle
                self.last_modified = datetime.now()
                return True
        return False
    
    def set_playback_speed(self, speed: float):
        """Set the playback speed (0.25x to 4x)"""
        if 0.25 <= speed <= 4.0:
            self.playback_speed = speed
            self.last_modified = datetime.now()
            return True
        return False
    
    def add_user(self, user_id: str):
        """Add a user to the active users list"""
        if user_id not in self.active_users:
            self.active_users.append(user_id)
            self.last_modified = datetime.now()
    
    def remove_user(self, user_id: str):
        """Remove a user from the active users list"""
        if user_id in self.active_users:
            self.active_users.remove(user_id)
            self.last_modified = datetime.now()
    
    def save_version(self, user_id: str, description: str) -> VideoVersion:
        """Save current state as a new version"""
        self.current_version += 1
        version = VideoVersion(
            version_id=str(uuid.uuid4()),
            version_number=self.current_version,
            created_by=user_id,
            created_at=datetime.now(),
            description=description,
            video_state=self._get_state_snapshot(),
            is_current=True
        )
        
        # Mark previous version as not current
        for v in self.versions:
            v.is_current = False
        
        self.versions.append(version)
        self.last_modified = datetime.now()
        return version
    
    def revert_to_version(self, version_number: int) -> bool:
        """Revert to a previous version"""
        for version in self.versions:
            if version.version_number == version_number:
                self._restore_state(version.video_state)
                return True
        return False
    
    def _get_state_snapshot(self) -> Dict:
        """Get a snapshot of the current video state"""
        return {
            'title': self.title,
            'description': self.description,
            'video_file': self.video_file,
            'duration': self.duration,
            'playback_speed': self.playback_speed,
            'subtitles': [s.to_dict() for s in self.subtitles],
            'current_version': self.current_version
        }
    
    def _restore_state(self, state: Dict):
        """Restore video state from a snapshot"""
        self.title = state['title']
        self.description = state['description']
        self.video_file = state['video_file']
        self.duration = state['duration']
        self.playback_speed = state['playback_speed']
        self.subtitles = [Subtitle.from_dict(s) for s in state['subtitles']]
        self.current_version = state['current_version']
        self.last_modified = datetime.now()


class RealTimeSyncManager:
    """Manages real-time synchronization between users"""
    
    def __init__(self):
        self.connected_users: Dict[str, User] = {}
        self.project_users: Dict[str, List[str]] = {}  # project_id -> list of user_ids
        self.edit_queue: List[Dict] = []
        self.message_queue: List[ChatMessage] = []
    
    def connect_user(self, user: User):
        """Connect a user to the system"""
        user.set_online()
        self.connected_users[user.user_id] = user
    
    def disconnect_user(self, user_id: str):
        """Disconnect a user from the system"""
        if user_id in self.connected_users:
            self.connected_users[user_id].set_offline()
            del self.connected_users[user_id]
    
    def join_project(self, user_id: str, project_id: str):
        """Add a user to a project"""
        if user_id not in self.connected_users:
            return False
        
        if project_id not in self.project_users:
            self.project_users[project_id] = []
        
        if user_id not in self.project_users[project_id]:
            self.project_users[project_id].append(user_id)
        
        return True
    
    def leave_project(self, user_id: str, project_id: str):
        """Remove a user from a project"""
        if project_id in self.project_users and user_id in self.project_users[project_id]:
            self.project_users[project_id].remove(user_id)
            return True
        return False
    
    def broadcast_edit(self, project_id: str, edit_data: Dict):
        """Broadcast an edit to all users in a project"""
        if project_id in self.project_users:
            for user_id in self.project_users[project_id]:
                if user_id in self.connected_users:
                    self.edit_queue.append({
                        'project_id': project_id,
                        'edit_data': edit_data,
                        'timestamp': datetime.now()
                    })
    
    def broadcast_message(self, project_id: str, message: ChatMessage):
        """Broadcast a chat message to all users in a project"""
        if project_id in self.project_users:
            for user_id in self.project_users[project_id]:
                if user_id in self.connected_users:
                    self.message_queue.append(message)
    
    def get_active_users(self, project_id: str) -> List[str]:
        """Get list of active users in a project"""
        return self.project_users.get(project_id, [])
    
    def get_pending_edits(self, project_id: str) -> List[Dict]:
        """Get pending edits for a project"""
        return [e for e in self.edit_queue if e['project_id'] == project_id]
    
    def get_pending_messages(self, project_id: str) -> List[ChatMessage]:
        """Get pending messages for a project"""
        return [m for m in self.message_queue if m.user_id in self.project_users.get(project_id, [])]


class SubtitleSynchronizer:
    """Handles automatic and manual subtitle synchronization"""
    
    def __init__(self):
        self.sync_history: List[Dict] = []
    
    def auto_sync_subtitles(self, video_duration: float, subtitle_file: str) -> List[Subtitle]:
        """
        Automatically synchronize subtitles with video content
        This is a simulation - in real implementation would use speech recognition
        """
        # Simulate subtitle generation based on video duration
        subtitles = []
        num_subtitles = int(video_duration / 5)  # One subtitle every 5 seconds
        
        for i in range(num_subtitles):
            start_time = i * 5
            end_time = min((i + 1) * 5, video_duration)
            
            subtitle = Subtitle(
                subtitle_id=str(uuid.uuid4()),
                start_time=start_time,
                end_time=end_time,
                text=f"Subtitle line {i + 1} - Auto generated",
                language="en",
                is_manual=False
            )
            subtitles.append(subtitle)
        
        self.sync_history.append({
            'type': 'auto_sync',
            'timestamp': datetime.now(),
            'subtitle_count': len(subtitles)
        })
        
        return subtitles
    
    def manual_adjust_subtitle(self, subtitle: Subtitle, 
                               new_start: float, new_end: float) -> Subtitle:
        """Manually adjust subtitle timing"""
        subtitle.start_time = new_start
        subtitle.end_time = new_end
        subtitle.is_manual = True
        
        self.sync_history.append({
            'type': 'manual_adjust',
            'timestamp': datetime.now(),
            'subtitle_id': subtitle.subtitle_id,
            'old_start': subtitle.start_time,
            'old_end': subtitle.end_time,
            'new_start': new_start,
            'new_end': new_end
        })
        
        return subtitle
    
    def align_subtitles(self, subtitles: List[Subtitle], 
                       offset: float) -> List[Subtitle]:
        """Apply a time offset to all subtitles"""
        for subtitle in subtitles:
            subtitle.start_time += offset
            subtitle.end_time += offset
            subtitle.is_manual = True
        
        self.sync_history.append({
            'type': 'align_all',
            'timestamp': datetime.now(),
            'offset': offset,
            'subtitle_count': len(subtitles)
        })
        
        return subtitles
    
    def get_sync_history(self) -> List[Dict]:
        """Get the synchronization history"""
        return self.sync_history


class ChatManager:
    """Manages real-time chat functionality"""
    
    def __init__(self):
        self.project_chats: Dict[str, List[ChatMessage]] = {}
    
    def send_message(self, project_id: str, user_id: str, 
                    username: str, content: str, 
                    message_type: str = "text") -> ChatMessage:
        """Send a chat message"""
        message = ChatMessage(
            message_id=str(uuid.uuid4()),
            user_id=user_id,
            username=username,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type
        )
        
        if project_id not in self.project_chats:
            self.project_chats[project_id] = []
        
        self.project_chats[project_id].append(message)
        return message
    
    def get_chat_history(self, project_id: str) -> List[ChatMessage]:
        """Get chat history for a project"""
        return self.project_chats.get(project_id, [])
    
    def send_suggestion(self, project_id: str, user_id: str, 
                       username: str, content: str) -> ChatMessage:
        """Send a suggestion message"""
        return self.send_message(project_id, user_id, username, 
                                content, message_type="suggestion")
    
    def send_feedback(self, project_id: str, user_id: str, 
                     username: str, content: str) -> ChatMessage:
        """Send a feedback message"""
        return self.send_message(project_id, user_id, username, 
                                content, message_type="feedback")


class VersionControlManager:
    """Manages version control for video projects"""
    
    def __init__(self):
        self.version_history: Dict[str, List[VideoVersion]] = {}
    
    def save_version(self, project_id: str, version: VideoVersion):
        """Save a new version"""
        if project_id not in self.version_history:
            self.version_history[project_id] = []
        
        self.version_history[project_id].append(version)
    
    def get_versions(self, project_id: str) -> List[VideoVersion]:
        """Get all versions for a project"""
        return self.version_history.get(project_id, [])
    
    def get_version(self, project_id: str, version_number: int) -> Optional[VideoVersion]:
        """Get a specific version"""
        versions = self.get_versions(project_id)
        for version in versions:
            if version.version_number == version_number:
                return version
        return None
    
    def compare_versions(self, project_id: str, 
                        version1: int, version2: int) -> Dict:
        """Compare two versions and return differences"""
        v1 = self.get_version(project